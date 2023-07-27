import numpy as np
import affinity as af
from edge import Edge

def create_affinity_matrix(graph1, graph2):
    nodes1 = graph1.get_nodes()
    nodes2 = graph1.get_nodes()
    
    M = np.zeros((len(nodes1) * len(nodes2), len(nodes1) * len(nodes2)))

    for i in range(len(nodes1)):
        for i_prime in range(len(nodes2)):
            for j in range(len(nodes1)):
                for j_prime in range(len(nodes2)):
                    if(i == j and i_prime == j_prime):
                        M[i * i_prime, j * j_prime] = af.calculate_node_similarity(nodes1[i], nodes2[j])
                    else:
                        M[i * i_prime, j * j_prime] = af.calculate_edge_similarity(Edge(nodes1[i], nodes1[j]), Edge(nodes2[i_prime], nodes2[j_prime]))

    return M

