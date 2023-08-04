import numpy as np
import scipy.optimize as opt

from matrix import create_affinity_matrix

def find_optimal_matching(w, n, m, threshold=0.5):
    G = np.ndarray((n, m))
    for i in range(n):
        for j in range(m):
            G[i][j] = w[(i * m) + j]
    rows, cols = opt.linear_sum_assignment(-np.array(G))
    matching_indices = [[rows[i], cols[i]] for i in range(len(rows)) if G[rows[i]][cols[i]] >= threshold]
    return matching_indices
