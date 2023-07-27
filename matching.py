from matrix import create_affinity_matrix
from utils import threshold_matching_results

# 这部分匈牙利算法代码参考自 https://oi-wiki.org/graph/graph-matching/bigraph-weight-match/ 
class Hungarian:
    def __init__(self, _n, _m):
        self.org_n = _n
        self.org_m = _m
        self.n = max(_n, _m)
        self.inf = float('inf')
        self.res = 0
        self.g = [[0]*self.n for _ in range(self.n)]
        self.matchx = [-1]*self.n
        self.matchy = [-1]*self.n
        self.pre = [0]*self.n
        self.visx = [False]*self.n
        self.visy = [False]*self.n
        self.lx = [-self.inf]*self.n
        self.ly = [0]*self.n
        self.slack = [0]*self.n

    def add_edge(self, u, v, w):
        self.g[u][v] = max(w, 0)

    def check(self, v):
        self.visy[v] = True
        if self.matchy[v] != -1:
            self.q.append(self.matchy[v])
            self.visx[self.matchy[v]] = True
            return False
        while v != -1:
            self.matchy[v] = self.pre[v]
            v, self.matchx[self.pre[v]] = self.matchx[self.pre[v]], v
        return True

    def bfs(self, i):
        self.q = [i]
        self.visx[i] = True
        while True:
            while self.q:
                u = self.q.pop(0)
                for v in range(self.n):
                    if not self.visy[v]:
                        delta = self.lx[u] + self.ly[v] - self.g[u][v]
                        if self.slack[v] >= delta:
                            self.pre[v] = u
                            if delta == 0 and self.check(v):
                                return
                            else:
                                self.slack[v] = delta
            a = min(self.slack[j] for j in range(self.n) if not self.visy[j])
            for j in range(self.n):
                if self.visx[j]:
                    self.lx[j] -= a
                if self.visy[j]:
                    self.ly[j] += a
                else:
                    self.slack[j] -= a
            for j in range(self.n):
                if not self.visy[j] and self.slack[j] == 0 and self.check(j):
                    return

    def solve(self):
        for i in range(self.n):
            self.lx[i] = max(self.g[i])
        for i in range(self.n):
            self.slack = [self.inf]*self.n
            self.visx = [False]*self.n
            self.visy = [False]*self.n
            self.bfs(i)
        for i in range(self.n):
            if self.g[i][self.matchx[i]] > 0:
                self.res += self.g[i][self.matchx[i]]
            else:
                self.matchx[i] = -1
        print(self.res)
        for i in range(self.org_n):
            print(self.matchx[i] + 1, end=' ')
        print()

def find_optimal_matching(M, w, n, m):
    h = Hungarian(n, m)
    for i in range(n):
        for j in range(m):
            h.add_edge(i, j, w[i*j])
            h.add_edge(j, i, w[i*j])
    h.solve()
