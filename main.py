import numpy as np
from scipy.optimize import minimize
import os
import jsonlines
import ast
import numba as nb

from match_seq import construct_similarity_tree
from tqdm import tqdm
from matrix import create_affinity_matrix
from matching import find_optimal_matching

def main(car1, car2):
    # Step 0: measure L1 and L2 (length)
    L1 = len(car1['category'])
    L2 = len(car2['category'])

    # Step 1: Construct information matrices
    G1 = np.zeros((L1, 11), dtype=np.float32)
    G2 = np.zeros((L2, 11), dtype=np.float32)
    # [0: category, position: 1, 2, 3, bounding_box: 4, 5, 6, world_position: 7, 8, 9, heading: 10]

    # Step 2: Define node and edge attributes
    for i in range(L1):
        G1[i]=[car1['category'][i], car1['position'][i][0], car1['position'][i][1], car1['position'][i][2], 
                    car1['bounding_box'][i][0], car1['bounding_box'][i][1], car1['bounding_box'][i][2],
                    car1['world_position'][i][0], car1['world_position'][i][1], car1['world_position'][i][2],
                    car1['heading'][i][0]]
        
    for i in range(L2):
        G2[i]=[car2['category'][i], car2['position'][i][0], car2['position'][i][1], car2['position'][i][2], 
                    car2['bounding_box'][i][0], car2['bounding_box'][i][1], car2['bounding_box'][i][2],
                    car2['world_position'][i][0], car2['world_position'][i][1], car2['world_position'][i][2],
                    car2['heading'][i][0]]

    # Step 3: Create affinity matrix
    M = create_affinity_matrix(G1, G2, L1, L2)
    
    w = np.zeros(len(M))

    # 定义目标函数
    @nb.njit()
    def objective_function(x=np.array([]), A=np.array([[]])):
        return -(x.T @ A @ x)

    # 定义约束条件：||x||^2 - 1 = 0
    @nb.njit()
    def constraint(x=np.array([])):
        return np.linalg.norm(x)**2 - 1

    constraint_eq = {'type': 'eq', 'fun': constraint}
    w_a_sol = minimize(objective_function, w, args=(M,), method='SLSQP', constraints=constraint_eq)
    # print(w_a_sol)

    w_a = w_a_sol.x
    w_a -= np.min(w_a)
    w_a /= np.max(w_a)

    # print(w_a)
    # print(np.where(w_a > 0.9))

    # Step 4: Solve graph matching problem
    matching_results = find_optimal_matching(w_a, L1, L2, threshold=0.5)
    # print(matching_results)

    return matching_results

def transformed_boxes(car_from_global, ref_from_car, pred_boxes):
    # 将前三个位置和第七个位置分别提取出来
    count = len(pred_boxes)
    positions = pred_boxes[:, :3]
    rotation = pred_boxes[:, 6]
    velocity = pred_boxes[:, 7:9]

    # 创建齐次坐标系表示的位置、旋转、速度信息
    positions_homogeneous = np.concatenate((positions, np.ones((len(pred_boxes), 1))), axis=1)
    rotation_homogeneous = np.stack(
        [np.cos(rotation), np.sin(rotation), np.zeros_like(rotation), np.zeros_like(rotation)],
        axis=1)
    velocity_homogeneous = np.concatenate((velocity, np.zeros((len(pred_boxes), 1)), np.ones((len(pred_boxes), 1))),
                                          axis=1)

    # 将位置和旋转信息合并为变换前的齐次坐标
    pred_boxes_homogeneous = np.concatenate((positions_homogeneous, rotation_homogeneous, velocity_homogeneous), axis=0)

    # 进行坐标变换
    transformed_boxes_homogeneous = np.matmul(np.linalg.inv(car_from_global),
                                              np.matmul(np.linalg.inv(ref_from_car), pred_boxes_homogeneous.T)).T

    # 提取变换后的位置和旋转信息
    transformed_positions = transformed_boxes_homogeneous[:count, :3]
    transformed_rotation = np.arctan2(transformed_boxes_homogeneous[count:count * 2, 1],
                                      transformed_boxes_homogeneous[count:count * 2, 0])
    transformed_velocity = transformed_boxes_homogeneous[count * 2:count * 3, :2]

    # 将位置和旋转信息合并为变换后的坐标
    transformed_boxes = np.concatenate(
        (transformed_positions, pred_boxes[:, 3:6], transformed_rotation[:, np.newaxis], transformed_velocity), axis=1)

    return transformed_boxes

if __name__ == '__main__':
    # incorrect_matching = 0
    # undiscoverd_matching = 0
    correct_matching = 0
    total_matching = 0
    boxes_count = 0
    covisible_boxes_count = 0

    in_path_pair = "./new_sweeps/val.json"
    pair_data = []
    with jsonlines.open(in_path_pair) as pairs:
        pair_data = [ast.literal_eval(pair) for pair in list(pairs)]

    pair_map = dict()
    for i in range(len(pair_data)):
        pair_map[pair_data[i]['file'] + ", " + pair_data[i]['src'] + ", " + pair_data[i]['dest']] = pair_data[i]['match_matrix']

    cnt = 0

    for filename in tqdm(os.listdir('./new_sweeps/')):
        
    #     print(f"Now processing {filename}")
        data = np.load(os.path.join("./new_sweeps/", filename), allow_pickle=True).item()
        cars = []

        simi = data['simi_matrix']
        matrix_simi = simi + simi.T
        simi_tree = construct_similarity_tree(0, matrix_simi)

        for key, value in data.items():
            if key.startswith('id_'):
                transformed_boxes_ret = transformed_boxes(value['car_from_global'], value['ref_from_car'], value['pred_boxes'])
                cars.append({'id': key, 'category': value['pred_labels'], 
                    'bounding_box': value['pred_boxes'][:, [3, 4, 5]], 'position': value['pred_boxes'][:, [0, 1, 2]],
                    'world_position': transformed_boxes_ret[:, [0, 1, 2]], 'heading': transformed_boxes_ret[:, [6]]})

        for pair in simi_tree:
            src, dest = pair[0], pair[1]
            key = filename + ", " + (f"id_{src}") + ", " + (f"id_{dest}")
            if key in pair_map:
                matrix = pair_map[key]
            else:
                matrix = None

            matching = main(cars[src], cars[dest])
            gt_matching = np.array(np.where(np.array(matrix) == 1)).transpose()
            # print(matching, '\n', gt_matching)

            valid = 0

            for ii in range(len(gt_matching)):
                found = False
                for jj in range(len(matching)):
                    if np.array_equal(gt_matching[ii], matching[jj]):
                        found = True
                if found:
                    valid += 1

            if len(gt_matching) >= valid >= 2:
                correct_matching += 1


            total_matching += 1
            # print(correct_matching, ' ', total_matching)

            covisible_boxes_count += np.sum(matrix)
            boxes_count += len(cars[src]) + len(cars[dest])

            # print(f"Current co-visible rate: {np.sum(matrix) / (len(boxes[i]) + len(boxes[j]))}")

        if cnt % 10 == 0:
            print(f"Round {cnt} status:")
            print(f"- Average co-visible rate: {(covisible_boxes_count / boxes_count) * 100:.2f}%")
            print(f"- Correct matching: {correct_matching}")
            print(f"- Total matching: {total_matching}")
            print(f"- Accuracy: {(correct_matching / total_matching) * 100:.2f}%")
        cnt += 1

    # print(f"Incorrect matching: {incorrect_matching}")
    # print(f"Undiscoverd matching: {undiscoverd_matching}")
    print("Final result:")
    print(f"- Average co-visible rate: {(covisible_boxes_count / boxes_count) * 100:.2f}%")
    print(f"- Correct matching: {correct_matching}")
    print(f"- Total matching: {total_matching}")
    print(f"- Accuracy: {(correct_matching / total_matching) * 100:.2f}%")

