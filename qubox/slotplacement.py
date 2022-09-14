import numpy as np
from qubox.base import Base

class SlotPlacement(Base):
    def __init__(self,
                wire_matrix,
                distance_matrix,
                ALPHA=1,
                BETA=1
                ):
        # Check tye type of Arguments
        if isinstance(wire_matrix, list):
            wire_matrix = np.array(wire_matrix)
        elif isinstance(wire_matrix, np.ndarray):
            pass
        else:
            print("The type of the argument 'wire_matrix' is WRONG.")
            print("It shoud be list/numpy.ndarray.")
            exit()
        if isinstance(distance_matrix, list):
            distance_matrix = np.array(distance_matrix)
        elif isinstance(distance_matrix, np.ndarray):
            pass
        else:
            print("The type of the argument 'distance_matrix' is WRONG.")
            print("It shoud be list/numpy.ndarray.")
            exit()

        NUM_ITEM = len(wire_matrix)
        NUM_SLOT = len(distance_matrix)
        self.wire_matrix = wire_matrix
        self.distance_matrix = distance_matrix
        super().__init__(NUM_SPIN = NUM_ITEM * NUM_SLOT)
        self.spin_index = np.arange(NUM_ITEM * NUM_SLOT).reshape(NUM_ITEM, NUM_SLOT)
        np.set_printoptions(edgeitems=10) # Chenge the setting for printing numpy

        self.cost_term(NUM_FACTORY)
        self.penalty_term(NUM_FACTORY, ALPHA)
        self.all_term()
        self.make_qubo_list()

    def cost_term(self, NUM_ITEM, NUM_SLOT):
        # Quadratic term
        for a in range(NUM_SLOT):
            for i in range(NUM_ITEM):
                for b in range(NUM_SLOT):
                    for j in range(NUM_ITEM):
                        idx_i = self.spin_index[i, a]
                        idx_j = self.spin_index[j, b]
                        coef = self.wire_matrix[i, j] * self.distance_matrix[a, b] / 2
                        if coef == 0:
                            continue
                        self.qubo_cost[idx_i, idx_j] += coef
        # Make QUBO upper triangular matrix
        self.qubo_cost = np.triu(self.qubo_cost + np.tril(self.qubo_cost, k=-1).T + np.triu(self.qubo_cost).T) - np.diag(self.qubo_cost.diagonal())

    def penalty_term(self, NUM_ITEM, NUM_SLOT, ALPHA):
        # Calculate constraint term1: item assignment constraint
        # (1-hot constraint of spin-matrix vertical line)
        # Quadratic term
        for i in range(NUM_ITEM):
            for a in range(NUM_SLOT-1):
                for b in range(a+1, NUM_SLOT):
                    idx_i = self.spin_index[i, a]
                    idx_j = self.spin_index[i, b]
                    coef = 2
                    self.qubo_penalty[idx_i, idx_j] += ALPHA * coef
        # Linear term
        for i in range(NUM_ITEM):
            for a in range(NUM_SLOT):
                idx = self.spin_index[i, a]
                coef = -1
                self.qubo_penalty[idx, idx] += ALPHA * coef
        # Constant term
        self.const_penalty[0] = ALPHA * NUM_ITEM

        # Calculate constraint term2: slot assignment constraint
        # (Constraint of spin-matrix horizontal line)
        # Quadratic term
        for a in range(NUM_SLOT):
            for i in range(NUM_ITEM-1):
                for j in range(i+1, NUM_ITEM):
                    idx_i = self.spin_index[i, a]
                    idx_j = self.spin_index[j, a]
                    coef = 2
                    self.qubo_penalty[idx_i, idx_j] += ALPHA * coef
