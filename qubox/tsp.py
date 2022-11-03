import numpy as np
from qubox.base import BaseQUBO

class TSP(BaseQUBO):
    def __init__(self,
                dist_mtx,
                ALPHA=1
                ):
        # Check tye type of Arguments
        if isinstance(dist_mtx, list):
            dist_mtx = np.array(dist_mtx)
        elif isinstance(dist_mtx, np.ndarray):
            pass
        else:
            print("The type of the argument 'dist_mtx' is WRONG.")
            print("It shoud be list/numpy.ndarray.")
            exit()

        NUM_CITY = len(dist_mtx)
        self.dist_mtx = dist_mtx
        super().__init__(num_spin = NUM_CITY * NUM_CITY)
        self.spin_index = np.arange(NUM_CITY * NUM_CITY).reshape(NUM_CITY, NUM_CITY)

        self.h_cost(NUM_CITY)
        self.h_pen(NUM_CITY, ALPHA)
        self.h_all()

    def h_cost(self, NUM_CITY):
        # Quadratic term
        for t in range(NUM_CITY):
            for u in range(NUM_CITY):
                for v in range(NUM_CITY):
                    if t < NUM_CITY-1:
                        idx_i = self.spin_index[t, u]
                        idx_j = self.spin_index[t+1, v]
                    elif t == NUM_CITY-1:
                        idx_i = self.spin_index[t, u]
                        idx_j = self.spin_index[0, v]
                    coef  = self.dist_mtx[u, v]
                    if coef == 0:
                        continue
                    self.Q_cost[idx_i, idx_j] += coef
        # Make QUBO upper triangular matrix
        self.Q_cost = np.triu(self.Q_cost) + np.tril(self.Q_cost).T - np.diag(self.Q_cost.diagonal())

    def h_pen(self, NUM_CITY, ALPHA):
        # Calculate constraint term1 (1-hot of horizontal line)
        # Quadratic term
        for t in range(NUM_CITY):
            for u in range(NUM_CITY-1):
                for v in range(u+1, NUM_CITY):
                    idx_i = self.spin_index[t, u]
                    idx_j = self.spin_index[t, v]
                    coef = 2
                    self.Q_pen[idx_i, idx_j] += ALPHA * coef
        # Linear term
        for t in range(NUM_CITY):
            for u in range(NUM_CITY):
                idx = self.spin_index[t, u]
                coef = -1
                self.Q_pen[idx, idx] += ALPHA * coef
        # Constant term
        self.const_pen += ALPHA * NUM_CITY

        # Calculate constraint term2 (1-hot of vertical line)
        # Quadratic term
        for u in range(NUM_CITY):
            for t in range(NUM_CITY-1):
                for tt in range(t+1, NUM_CITY):
                    idx_i = self.spin_index[t, u]
                    idx_j = self.spin_index[tt, u]
                    coef = 2
                    self.Q_pen[idx_i, idx_j] += ALPHA * coef
        # Linear term
        for u in range(NUM_CITY):
            for t in range(NUM_CITY):
                idx = self.spin_index[t, u]
                coef = -1
                self.Q_pen[idx, idx] += ALPHA * coef
        # Constant term
        self.const_pen += ALPHA * NUM_CITY
