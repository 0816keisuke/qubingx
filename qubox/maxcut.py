import numpy as np
from qubox.base import Base

class MaxCut(Base):
    def __init__(self, adjacency_matrix):
        # Check tye type of Arguments
        if isinstance(adjacency_matrix, list):
            adjacency_matrix = np.array(adjacency_matrix)
        elif isinstance(adjacency_matrix, np.ndarray):
            pass
        else:
            print("The type of the argument 'adjacency_matrix' is WRONG.")
            print("It shoud be list/numpy.ndarray.")
            exit()

        NUM_VERTEX = len(adjacency_matrix)
        super().__init__(num_spin = NUM_VERTEX)
        self.adjacency_matrix = adjacency_matrix
        np.set_printoptions(edgeitems=10) # Chenge the setting for printing numpy

        self.cost_term(NUM_VERTEX)
        self.penalty_term()
        self.all_term()

    def cost_term(self, NUM_VERTEX):
        for i in range(NUM_VERTEX-1):
            for j in range(i+1, NUM_VERTEX):
                w_ij = self.adjacency_matrix[i, j]
                self.qubo_cost[i, j] += 2 * w_ij
                self.qubo_cost[i, i] += -1 * w_ij
                self.qubo_cost[j, j] += -1 * w_ij

    def penalty_term(self):
        pass
