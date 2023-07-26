# main.py

from graph import Graph
from node import Node
from edge import Edge
from affinity import calculate_node_affinity, calculate_edge_affinity
from matrix import create_affinity_matrix
from matching import find_optimal_matching
from utils import threshold_matching_results

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

def main(car1, car2):
    # Step 1: Construct graphs G𝑉 and G𝐼
    G𝑉 = Graph()
    G𝐼 = Graph()

    # Step 2: Define node and edge attributes
    # ...

    # Step 3: Calculate node and edge affinity
    calculate_node_affinity(G𝑉)
    calculate_edge_affinity(G𝑉)

    # Step 4: Create affinity matrix
    M = create_affinity_matrix(G𝑉)

    # Step 5: Solve graph matching problem
    matching_results = find_optimal_matching(M)

    # Step 6: Threshold matching results
    thresholded_results = threshold_matching_results(matching_results, threshold=0.5)

    # Print the final matching results
    for node_pair in thresholded_results:
        print(f"Matched nodes: {node_pair[0].id} and {node_pair[1].id}")

    return thresholded_results

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
                boxes.append(value['pred_boxes'])
                transformed_boxes_ret = transformed_boxes(value['car_from_global'], value['ref_from_car'], value['pred_boxes'])
                world_boxes.append(transformed_boxes_ret[:, [0, 1, 2, 4, 5, 3, 6]])
                cars.append({id: key, categories: value['pred_labels'], 
                    bounding_boxes: value['pred_boxes'][:, [3, 4, 5]], positions: value['pred_boxes'][:, [0, 1, 2]],
                    world_positions: transformed_boxes_ret[:, [0, 1, 2]], headings: transformed_boxes_ret[:, [6]]})
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

        print(f"Incoorrect matching: {incorrect_matching}")
        print(f"Undiscoverd matching: {undiscoverd_matching}")
        print(f"Total matching: {total_matching}")
