import numpy as np
from qubox.base import Base

class MaxCut(Base):
    def __init__(self, adjacency_mtx):
        # Check tye type of Arguments
        if isinstance(adjacency_mtx, list):
            adjacency_mtx = np.array(adjacency_mtx)
        elif isinstance(adjacency_mtx, np.ndarray):
            pass
        else:
            print("The type of the argument 'adjacency_mtx' is WRONG.")
            print("It shoud be list/numpy.ndarray.")
            exit()

        NUM_VERTEX = len(adjacency_mtx)
        super().__init__(num_spin = NUM_VERTEX)
        self.adjacency_mtx = adjacency_mtx
        np.set_printoptions(edgeitems=10) # Chenge the setting for printing numpy

        self.h_cost(NUM_VERTEX)
        self.h_pen()
        self.h_all()

    def h_cost(self, NUM_VERTEX):
        for i in range(NUM_VERTEX-1):
            for j in range(i+1, NUM_VERTEX):
                w_ij = self.adjacency_mtx[i, j]
                self.q_cost[i, j] += 2 * w_ij
                self.q_cost[i, i] += -1 * w_ij
                self.q_cost[j, j] += -1 * w_ij

    def h_pen(self):
        pass
