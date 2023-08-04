import numpy as np
from numba import jit
import numba as nb


# M[i][i_prime][i][i_prime]
# @jit(fastmath=True)
def calculate_node_similarity(node1, node2):
    # Calculate node similarity based on attributes
    lambda_1 = 0.5
    lambda_2 = 0.1
    # according to original paper, they set them pirically 0.5 and 0.1 and we do as so
    f_1_i_ip = (node1.category == node2.category)
    f_2_i_ip = np.exp(-lambda_1 * np.square(np.linalg.norm(node1.bounding_box - node2.bounding_box)))
    # f_3_i_ip = (in our case, the trajectory affinity is not available since other algorithm doesnt have velocity data, so default set it to 1)
    f_3_i_ip = 1
    f_4_i_ip = np.exp(-lambda_2 * np.linalg.norm(node1.world_position - node2.world_position))

    miu_1 = 0.5
    miu_2 = 0.5
    # according to original paper, miu_1 and miu_2 are weights to balance those affinities and as they told that should all be 0.5
    return f_1_i_ip * f_3_i_ip * (miu_1 * f_2_i_ip + miu_2 * f_4_i_ip)


# M[i][i_prime][j][j_prime]
# @jit(fastmath=True)
def calculate_edge_similarity(e1n1, e1n2, e2n1, e2n2):
    # Calculate edge similarity based on attributes
    lambda_3 = 0.5
    lambda_4 = 0.1
    # according to original paper, they set them pirically 0.5 and 0.1 and we do as so
    g_1_i_ip_j_jp = (((e1n1.category == e2n1.category) and (
                e1n2.category == e2n2.category)) or ((
                                 e1n2.category == e2n1.category) and (
                                 e1n1.category == e2n2.category)))
    g_2_i_ip_j_jp = np.exp(
        -lambda_3 * (np.linalg.norm(e1n1.position - e1n2.position) - np.linalg.norm(
            e2n1.position - e2n2.position))**2)
    g_3_i_ip_j_jp = np.exp(-lambda_4 * np.abs(
        np.sin(e1n1.heading - e1n2.heading) - np.sin(
            e2n1.heading - e2n2.heading)))

    miu_3 = 0.5
    miu_4 = 0.5
    # according to original paper, miu_3 and miu_4 also are weights to balance those affinities and as they told that should all be 0.5
    return g_1_i_ip_j_jp * (miu_3 * g_2_i_ip_j_jp + miu_4 * g_3_i_ip_j_jp)
