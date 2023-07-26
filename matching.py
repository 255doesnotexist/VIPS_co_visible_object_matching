# matching.py

from matrix import create_affinity_matrix
from utils import threshold_matching_results

def find_optimal_matching(M):
    # TODO: Implement the algorithm to find the optimal matching using the affinity matrix M
    pass

def match_nodes(node1, node2):
    # TODO: Implement the function to calculate the affinity between two nodes
    pass

def match_edges(edge1, edge2):
    # TODO: Implement the function to calculate the affinity between two edges
    pass

def find_matching_results(G𝑉, threshold):
    # Step 4: Create affinity matrix
    M = create_affinity_matrix(G𝑉)

    # Step 5: Solve graph matching problem
    matching_results = find_optimal_matching(M)

    # Step 6: Threshold matching results
    thresholded_results = threshold_matching_results(matching_results, threshold)

    return thresholded_results
