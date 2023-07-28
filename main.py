import numpy as np
from scipy.optimize import minimize
import os

from graph import Graph
from node import Node
from edge import Edge
from affinity import calculate_node_similarity, calculate_edge_similarity
from matrix import create_affinity_matrix
from matching import find_optimal_matching
from utils import threshold_matching_results

def main(car1, car2):
    # Step 1: Construct graphs
    G1 = Graph()
    G2 = Graph()

    # Step 2: Define node and edge attributes
    for i in range(len(car1['category'])):
        G1.add_node(Node(i, category=car1['category'][i], position=car1['position'][i], 
                    bounding_box=car1['bounding_box'][i], world_position=car1['world_position'][i],
                    heading=car1['heading'][i]))
        
    for i in range(len(car2['category'])):
        G2.add_node(Node(i, category=car2['category'][i], position=car2['position'][i], 
                    bounding_box=car2['bounding_box'][i], world_position=car2['world_position'][i],
                    heading=car2['heading'][i]))

    # Step 3: Create affinity matrix
    M = create_affinity_matrix(G1, G2)
    
    w = np.ones(len(M))

    # 定义目标函数
    def objective_function(x, A):
        return -(x.T @ A @ x)

    # 定义约束条件：||x||^2 - 1 = 0
    def constraint(x):
        return np.linalg.norm(x)**2 - 1
    
    constraint_eq = {'type': 'eq', 'fun': constraint}
    w_a_sol = minimize(objective_function, w, args=(M,), method='SLSQP', constraints=constraint_eq)

    w_a = w_a_sol.x
    # w_a -= np.min(w_a)
    # w_a /= np.max(w_a)

    # print(w_a)

    # Step 4: Solve graph matching problem
    matching_results = find_optimal_matching(M, w_a, len(G1.get_nodes()), len(G2.get_nodes()), threshold=0.5)
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
    incorrect_matching = 0
    undiscoverd_matching = 0
    total_matching = 0

    for filename in os.listdir('./new_sweeps/'):
    # for filename in ['scene_100_000026.npy']:
        print(f"Now processing {filename}")
        data = np.load(os.path.join("./new_sweeps/", filename), allow_pickle=True).item()
        cars = []
        matrix_iou = None
        matrix_dis = None
        for key, value in data.items():
            if key.startswith('id_'):
                transformed_boxes_ret = transformed_boxes(value['car_from_global'], value['ref_from_car'], value['pred_boxes'])
                cars.append({'id': key, 'category': value['pred_labels'], 
                    'bounding_box': value['pred_boxes'][:, [3, 4, 5]], 'position': value['pred_boxes'][:, [0, 1, 2]],
                    'world_position': transformed_boxes_ret[:, [0, 1, 2]], 'heading': transformed_boxes_ret[:, [6]]})
            if key == 'matrix_iou':
                matrix_iou = value
            if key == 'matrix_dis':
                matrix_dis = value
        # print(cars)
        # print(matrix_iou)
        # print(matrix_dis)
        for i in range(len(cars)):
            for j in range(i + 1, len(cars)):
                if i != j:
                    matching = main(cars[i], cars[j])
                    if len(matching) > 0 and matrix_iou[i][j] == 0:
                        incorrect_matching += 1
                    if len(matching) == 0 and matrix_iou[i][j] > 0:
                        undiscoverd_matching += 1
                total_matching += 1

        print(f"Incorrect matching: {incorrect_matching}")
        print(f"Undiscoverd matching: {undiscoverd_matching}")
        print(f"Total matching: {total_matching}")
