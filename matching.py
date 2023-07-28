import numpy as np
import scipy.optimize as opt
import torch

from matrix import create_affinity_matrix
from utils import threshold_matching_results

# 手写匈牙利？感觉不如调库 
def _hung_kernel(s: torch.Tensor, n1=None, n2=None):
    if n1 is None:
        n1 = s.shape[0]
    if n2 is None:
        n2 = s.shape[1]
    row, col = opt.linear_sum_assignment(s[:n1, :n2])
    return row, col

def find_optimal_matching(M, w, n, m, threshold = 0.5):
    G = np.ndarray((n,m))
    for i in range(n):
        for j in range(m):
            G[i][j] = w[i * j]

    row, col = _hung_kernel(torch.from_numpy(G))
    
    perm_mat = np.zeros_like(G)

    for k in range(len(row)):
        if w[row[k] * col[k]] >= threshold:
            perm_mat[row[k], col[k]] = 1
    
    return perm_mat