import numpy as np

def create_affinity_matrix(graph):
    nodes = graph.get_nodes()
    num_nodes = len(nodes)
    M = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            node1 = nodes[i]
            node2 = nodes[j]
            edge = graph.get_edge_by_nodes(node1, node2)
            affinity = edge.get_affinity()
            M[i, j] = affinity
            M[j, i] = affinity

    return M

