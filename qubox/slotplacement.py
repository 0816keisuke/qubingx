import numpy as np
from qubox.base import BaseQUBO

class SlotPlacement(BaseQUBO):
    def __init__(self,
                wire_mtx,
                dist_mtx,
                ALPHA=1,
                BETA=1
                ):
        # Check tye type of Arguments
        if isinstance(wire_mtx, list):
            wire_mtx = np.array(wire_mtx)
        elif isinstance(wire_mtx, np.ndarray):
            pass
        else:
            print("The type of the argument 'wire_mtx' is WRONG.")
            print("It shoud be list/numpy.ndarray.")
            exit()
        if isinstance(dist_mtx, list):
            dist_mtx = np.array(dist_mtx)
        elif isinstance(dist_mtx, np.ndarray):
            pass
        else:
            print("The type of the argument 'dist_mtx' is WRONG.")
            print("It shoud be list/numpy.ndarray.")
            exit()

        NUM_ITEM = len(wire_mtx)
        NUM_SLOT = len(dist_mtx)
        self.wire_mtx = wire_mtx
        self.dist_mtx = dist_mtx
        super().__init__(num_spin = NUM_ITEM * NUM_SLOT)
        self.spin_index = np.arange(NUM_ITEM * NUM_SLOT).reshape(NUM_ITEM, NUM_SLOT)

        self.h_cost(NUM_ITEM, NUM_SLOT)
        self.h_pen(NUM_ITEM, NUM_SLOT, ALPHA, BETA)
        self.h_all()

    def h_cost(self, NUM_ITEM, NUM_SLOT):
        # Quadratic term
        for a in range(NUM_SLOT):
            for i in range(NUM_ITEM):
                for b in range(NUM_SLOT):
                    for j in range(NUM_ITEM):
                        idx_i = self.spin_index[i, a]
                        idx_j = self.spin_index[j, b]
                        coef = self.wire_mtx[i, j] * self.dist_mtx[a, b] / 2
                        if coef == 0:
                            continue
                        self.Q_cost[idx_i, idx_j] += coef
        # Make QUBO upper triangular matrix
        self.Q_cost = np.triu(self.Q_cost) + np.tril(self.Q_cost).T - np.diag(self.Q_cost.diagonal())

    def h_pen(self, NUM_ITEM, NUM_SLOT, ALPHA, BETA):
        # Calculate constraint term1: item assignment constraint
        # (1-hot constraint of spin-matrix vertical line)
        # Quadratic term
        for i in range(NUM_ITEM):
            for a in range(NUM_SLOT-1):
                for b in range(a+1, NUM_SLOT):
                    idx_i = self.spin_index[i, a]
                    idx_j = self.spin_index[i, b]
                    coef = 2
                    self.Q_pen[idx_i, idx_j] += ALPHA * coef
        # Linear term
        for i in range(NUM_ITEM):
            for a in range(NUM_SLOT):
                idx = self.spin_index[i, a]
                coef = -1
                self.Q_pen[idx, idx] += ALPHA * coef
        # Constant term
        self.const_pen = ALPHA * NUM_ITEM

        # Calculate constraint term2: slot assignment constraint
        # (Constraint of spin-matrix horizontal line)
        # Quadratic term
        for a in range(NUM_SLOT):
            for i in range(NUM_ITEM-1):
                for j in range(i+1, NUM_ITEM):
                    idx_i = self.spin_index[i, a]
                    idx_j = self.spin_index[j, a]
                    coef = 2
                    self.Q_pen[idx_i, idx_j] += BETA * coef
