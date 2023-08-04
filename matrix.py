from itertools import product

import numpy as np
import affinity as af
from edge import Edge
import sys
from numba import jit
import numba as nb

@jit(nopython=True)
def create_affinity_matrix(graph1, graph2):
    nodes1 = graph1.get_nodes()
    nodes2 = graph2.get_nodes()

    M = np.zeros((len(nodes1) * len(nodes2), len(nodes1) * len(nodes2)))
    len_nodes1 = len(nodes1)
    len_nodes2 = len(nodes2)

    for (i, j), (i_prime, j_prime) in product(product(range(len_nodes1), repeat=2), product(range(len_nodes2), repeat=2)):
            if (i == j and i_prime == j_prime):
                M[i * len(nodes2) + i_prime, j * len(nodes2) + j_prime] = af.calculate_node_similarity(
                    nodes1[i], nodes2[i_prime])
            else:
                M[i * len(nodes2) + i_prime, j * len(nodes2) + j_prime] = af.calculate_edge_similarity(
                    nodes1[i], nodes1[j], nodes2[i_prime], nodes2[j_prime])

    return M
