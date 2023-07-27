import numpy as np

# M[i][i_prime][i][i_prime]
def calculate_node_similarity(node1, node2):
    # Calculate node similarity based on attributes
    lambda_1 = 0.5
    lambda_2 = 0.1
    # according to original paper, they set them pirically 0.5 and 0.1 and we do as so
    f_1_i_ip = (node1.category == node2.category)
    f_2_i_ip = np.exp(-lambda_1*np.square(np.linalg.norm(node1.bounding_box-node2.bounding_box)))
    # f_3_i_ip = (in our case, the trajectory affinity is not available since other algorithm doesnt have velocity data, so default set it to 1)
    f_3_i_ip = 1
    f_4_i_ip = np.exp(-lambda_2*np.linalg.norm(node1.world_position - node2.world_position))

    miu_1 = 0.5
    miu_2 = 0.5
    # according to original paper, miu_1 and miu_2 are weights to balance those affinities and as they told that should all be 0.5
    return f_1_i_ip * f_3_i_ip * (miu_1 * f_2_i_ip + miu_2 * f_4_i_ip)

# M[i][i_prime][j][j_prime]
def calculate_edge_similarity(edge1, edge2):
    # Calculate edge similarity based on attributes
    lambda_3 = 0.5
    lambda_4 = 0.1
    # according to original paper, they set them pirically 0.5 and 0.1 and we do as so
    g_1_i_ip_j_jp = ((edge1.get_node1().category == edge2.get_node1().category) and (edge1.get_node2().category == edge2.get_node2().category))
    g_2_i_ip_j_jp = np.exp(-lambda_3 * np.linalg.norm(edge1.get_node1().position - edge1.get_node2().position) - np.linalg.norm(edge2.get_node1().position - edge2.get_node2().position))
    g_3_i_ip_j_jp = np.exp(-lambda_4 * np.linalg.norm(np.sin(edge1.get_node1().heading - edge1.get_node2().heading) - np.sin(edge2.get_node1().heading - edge2.get_node2().heading)))

    miu_3 = 0.5
    miu_4 = 0.5
    # according to original paper, miu_3 and miu_4 also are weights to balance those affinities and as they told that should all be 0.5
    return g_1_i_ip_j_jp * (miu_3 * g_2_i_ip_j_jp + miu_4 * g_3_i_ip_j_jp)
