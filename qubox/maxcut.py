import numpy as np

from qubox.base import Base


class MaxCut(Base):
    def __init__(self, adjacency_mtx, mtx="upper"):
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
        super().__init__(modeltype="QUBO", mtx=mtx, num_spin=NUM_VERTEX)
        self.__check_mtx_type__()
        self.adjacency_mtx = adjacency_mtx

        self.hamil_cost(NUM_VERTEX)
        self.hamil_pen()
        self.__upper2sym__() # Execute if mtx=="sym"
        self.hamil_all()

    def hamil_cost(self, NUM_VERTEX):
        for i in range(NUM_VERTEX - 1):
            for j in range(i + 1, NUM_VERTEX):
                w_ij = self.adjacency_mtx[i, j]
                self.Q_cost[i, j] += 2 * w_ij
                self.Q_cost[i, i] += -1 * w_ij
                self.Q_cost[j, j] += -1 * w_ij

    def hamil_pen(self):
        pass
