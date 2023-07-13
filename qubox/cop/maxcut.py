import numpy as np

from qubox.cop.base import Matrix, Model
from qubox.cop.qubo import QUBO
import numpy as np
from typing import List


class MaxCut(QUBO):
    def __init__(
        self,
        adjacency_mtx: List[List[float | int]] | np.ndarray,
        MODEL: str = "QUBO",
        MATRIX: str = "upper",
    ):
        if isinstance(adjacency_mtx, np.ndarray):
            adjacency_mtx = adjacency_mtx.tolist()

        NUM_VERTEX = len(adjacency_mtx)
        super().__init__(
            MODEL=Model(MODEL),
            MATRIX=Matrix(MATRIX),
            num_spin=NUM_VERTEX,
            q_all=np.zeros((NUM_VERTEX, NUM_VERTEX)),
            q_obj=np.zeros((NUM_VERTEX, NUM_VERTEX)),
            q_constraint=np.zeros((NUM_VERTEX, NUM_VERTEX)),
            const_all=0,
            const_obj=0,
            const_constraint=0,
        )

        self.h_obj(NUM_VERTEX, adjacency_mtx)
        self.h_constraint()
        self.h_all()

    def hamil_cost(self, NUM_VERTEX: int, adjacency_mtx: List[List[float]]):
        for i in range(NUM_VERTEX - 1):
            for j in range(i + 1, NUM_VERTEX):
                w_ij = self.adjacency_mtx[i, j]
                self.Q_cost[i, j] += 2 * w_ij
                self.Q_cost[i, i] += -1 * w_ij
                self.Q_cost[j, j] += -1 * w_ij

    def hamil_pen(self):
        pass
