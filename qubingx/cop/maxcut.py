import numpy as np
import numpy.typing as npt

from qubingx.cop.base import Base, Model


class MaxCut(Base):
    def __init__(
        self,
        adjacency_mtx: list[list[float | int]] | npt.NDArray,
    ):
        if isinstance(adjacency_mtx, np.ndarray):
            adjacency_mtx = adjacency_mtx.tolist()

        num_vertex = len(adjacency_mtx)
        super().__init__(
            model=Model("QUBO"),
            num_spin=num_vertex,
            h_all=np.zeros((num_vertex, num_vertex)),
            h_obj=np.zeros((num_vertex, num_vertex)),
            h_constraint=np.zeros((num_vertex, num_vertex)),
            const_all=0,
            const_obj=0,
            const_constraint=0,
        )

        self._make_h_obj(num_vertex, adjacency_mtx)
        self._make_h_constraint()
        self._make_h_all()

    def _make_h_obj(self, num_vertex: int, adjacency_mtx: list[list[float]]):
        for i in range(num_vertex - 1):
            for j in range(i + 1, num_vertex):
                w_ij = adjacency_mtx[i][j]
                self.h_obj[i, j] += 2 * w_ij
                self.h_obj[i, i] += -1 * w_ij
                self.h_obj[j, j] += -1 * w_ij

    def _make_h_constraint(self):
        pass
