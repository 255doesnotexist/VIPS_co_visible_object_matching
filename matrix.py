from itertools import product

import numpy as np
import sys
from numba import jit
import numba as nb

# M[i][i_prime][i][i_prime]
@jit(nopython=True)
def calculate_node_similarity(N1=np.array([]), N2=np.array([])):
    # Calculate node similarity based on attributes
    lambda_1 = 0.5
    lambda_2 = 0.1
    # according to original paper, they set them pirically 0.5 and 0.1 and we do as so
    f_1_i_ip = (N1[0] == N2[0])
    f_2_i_ip = np.exp(-lambda_1 * np.square(np.linalg.norm(N1[4:6] - N2[4:6])))
    # f_3_i_ip = (in our case, the trajectory affinity is not available since other algorithm doesnt have velocity data, so default set it to 1)
    f_3_i_ip = 1
    f_4_i_ip = np.exp(-lambda_2 * np.linalg.norm(N1[7:9] - N2[7:9]))

    miu_1 = 0.5
    miu_2 = 0.5
    # according to original paper, miu_1 and miu_2 are weights to balance those affinities and as they told that should all be 0.5
    return f_1_i_ip * f_3_i_ip * (miu_1 * f_2_i_ip + miu_2 * f_4_i_ip)


# M[i][i_prime][j][j_prime]
@jit(nopython=True)
def calculate_edge_similarity(e1n1=np.array([]), e1n2=np.array([]), e2n1=np.array([]), e2n2=np.array([])):
    # Calculate edge similarity based on attributes
    lambda_3 = 0.5
    lambda_4 = 0.1
    # according to original paper, they set them pirically 0.5 and 0.1 and we do as so
    g_1_i_ip_j_jp = (((e1n1[0] == e2n1[0]) and (
                e1n2[0] == e2n2[0])) or ((
                                 e1n2[0] == e2n1[0]) and (
                                 e1n1[0] == e2n2[0])))
    g_2_i_ip_j_jp = np.exp(
        -lambda_3 * (np.linalg.norm(e1n1[1:3] - e1n2[1:3]) - np.linalg.norm(
            e2n1[1:3] - e2n2[1:3]))**2)
    g_3_i_ip_j_jp = np.exp(-lambda_4 * np.abs(
        np.sin(e1n1[10] - e1n2[10]) - np.sin(
            e2n1[10] - e2n2[10])))

    miu_3 = 0.5
    miu_4 = 0.5
    # according to original paper, miu_3 and miu_4 also are weights to balance those affinities and as they told that should all be 0.5
    return g_1_i_ip_j_jp * (miu_3 * g_2_i_ip_j_jp + miu_4 * g_3_i_ip_j_jp)


@jit(nopython=True)
def create_affinity_matrix(N1=np.array([[]]), N2=np.array([[]]), L1=np.float64, L2=np.float64):
    M = np.zeros((L1 * L2, L1 * L2))
    len_N1 = L1
    len_N2 = L2

    # for (i, j), (i_prime, j_prime) in product(product(range(len_N1), repeat=2), product(range(len_N2), repeat=2)):
    #         if (i == j and i_prime == j_prime):
    #             M[i * L2 + i_prime, j * L2 + j_prime] = af.calculate_node_similarity(
    #                 N1[i], N2[i_prime])
    #         else:
    #             M[i * L2 + i_prime, j * L2 + j_prime] = af.calculate_edge_similarity(
    #                 N1[i], N1[j], N2[i_prime], N2[j_prime])

    for i in range(len_N1):
        for j in range(len_N1):
            for i_prime in range(len_N2):
                for j_prime in range(len_N2):
    # for (i, j), (i_prime, j_prime) in product(product(range(len_N1), repeat=2), product(range(len_N2), repeat=2)):
                    if (i == j and i_prime == j_prime):
                        M[i * L2 + i_prime, j * L2 + j_prime] = calculate_node_similarity(
                            N1[i], N2[i_prime])
                    else:
                        M[i * L2 + i_prime, j * L2 + j_prime] = calculate_edge_similarity(
                            N1[i], N1[j], N2[i_prime], N2[j_prime])

    return M
