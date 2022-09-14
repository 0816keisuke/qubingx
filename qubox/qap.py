import numpy as np
from qubox.base import Base

class QAP(Base):
    def __init__(self,
                factory_matrix,
                distance_matrix,
                ALPHA=1
                ):
        # Check tye type of Arguments
        if isinstance(factory_matrix, list):
            factory_matrix = np.array(factory_matrix)
        elif isinstance(factory_matrix, np.ndarray):
            pass
        else:
            print("The type of the argument 'factory_matrix' is WRONG.")
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

        NUM_FACTORY = len(factory_matrix)
        self.factory_matrix = factory_matrix
        self.distance_matrix = distance_matrix
        super().__init__(NUM_SPIN = NUM_FACTORY * NUM_FACTORY)
        self.spin_index = np.arange(NUM_FACTORY * NUM_FACTORY).reshape(NUM_FACTORY, NUM_FACTORY)
        np.set_printoptions(edgeitems=10) # Chenge the setting for printing numpy

        self.cost_term(NUM_FACTORY)
        self.penalty_term(NUM_FACTORY, ALPHA)
        self.all_term()
        self.make_qubo_list()

    def cost_term(self, NUM_FACTORY):
        # Quadratic term
        for i in range(NUM_FACTORY):
            for j in range(NUM_FACTORY):
                for k in range(NUM_FACTORY):
                    for l in range(NUM_FACTORY):
                        idx_i = self.spin_index[i, k]
                        idx_j = self.spin_index[j, l]
                        coef = self.factory_matrix[i, j] * self.distance_matrix[k, l]
                        if coef == 0:
                            continue
                        self.qubo_cost[idx_i, idx_j] += coef
        # Make QUBO upper triangular matrix
        for i in range(self.NUM_SPIN):
            for j in range(i+1, self.NUM_SPIN):
                self.qubo_cost[j, i] = 0

    def penalty_term(self, NUM_FACTORY, ALPHA):
        # Constraint term1 (1-hot of horizontal line)
        # Quadratic term
        for i in range(NUM_FACTORY):
            for k in range(NUM_FACTORY-1):
                for l in range(k+1, NUM_FACTORY):
                    idx_i = self.spin_index[i, k]
                    idx_j = self.spin_index[i, l]
                    coef = 2
                    self.qubo_penalty[idx_i, idx_j] += ALPHA * coef
        # Linear term
        for i in range(NUM_FACTORY):
            for k in range(NUM_FACTORY):
                idx = self.spin_index[i, k]
                coef = -1
                self.qubo_penalty[idx, idx] += ALPHA * coef
        # Constant term
        self.const_penalty[0] += ALPHA * NUM_FACTORY

        # Constraint term2 (1-hot of vertical line)
        # Quadratic term
        for k in range(NUM_FACTORY):
            for i in range(NUM_FACTORY-1):
                for j in range(i+1, NUM_FACTORY):
                    idx_i = self.spin_index[i, k]
                    idx_j = self.spin_index[j, k]
                    coef = 2
                    self.qubo_penalty[idx_i, idx_j] += ALPHA * coef
        # Linear term
        for k in range(NUM_FACTORY):
            for i in range(NUM_FACTORY):
                idx = self.spin_index[i, k]
                coef = -1
                self.qubo_penalty[idx, idx] += ALPHA * coef
        # Constant term
        self.const_penalty[0] += ALPHA * NUM_FACTORY
