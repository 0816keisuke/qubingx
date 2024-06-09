from typing import Literal

import numpy as np
import numpy.typing as npt

from qubingx.cop.base import Base, Model


class GraphPartition(Base):
    def __init__(
        self,
        model_type: Literal["ISING", "QUBO"],
        adjacency_mtx: list[list[float | int]] | npt.NDArray,
        alpha: float = 1.0,
    ):
        if model_type not in ["ISING", "QUBO"]:
            raise ValueError("model_type must be 'ISING' or 'QUBO'")
        if isinstance(adjacency_mtx, np.ndarray):
            adjacency_mtx = adjacency_mtx.tolist()

        num_vertex = len(adjacency_mtx)
        super().__init__(
            model=Model("ISING") if model_type == "ISING" else Model("QUBO"),
            num_spin=num_vertex,
            h_all=np.zeros((num_vertex, num_vertex)),
            h_obj=np.zeros((num_vertex, num_vertex)),
            h_constraint=np.zeros((num_vertex, num_vertex)),
            const_all=0,
            const_obj=0,
            const_constraint=0,
        )

        self._make_h_obj(num_vertex, adjacency_mtx)
        self._make_h_constraint(num_vertex, adjacency_mtx, alpha)
        self._make_h_all()

    def _make_h_obj(self, num_vertex: int, adjacency_mtx: list[list[float]]):
        if self.model == Model.ISING:
            for i in range(num_vertex - 1):
                for j in range(i + 1, num_vertex):
                    w_ij = adjacency_mtx[i][j]
                    self.h_obj[i, j] += (-1) * w_ij / 2
                    self.const_obj += w_ij / 2
        elif self.model == Model.QUBO:
            for i in range(num_vertex):
                for j in range(i + 1, num_vertex):
                    if adjacency_mtx[i][j] == 0:
                        continue
                    self.h_obj[i, j] += -2 * adjacency_mtx[i][j]
                self.h_obj[i, i] += sum(adjacency_mtx[i])

    def _make_h_constraint(
        self, num_vertex: int, adjacency_mtx: list[list[float]], alpha: float = 1.0
    ):
        if self.model == Model.ISING:
            for i in range(num_vertex - 1):
                for j in range(i + 1, num_vertex):
                    self.h_constraint[i, j] += alpha * 2 * adjacency_mtx[i][j]
            self.const_constraint = alpha * num_vertex
        elif self.model == Model.QUBO:
            # Quadratic term
            for i in range(num_vertex - 1):
                for j in range(i + 1, num_vertex):
                    self.h_constraint[i, j] += 2 * alpha
            # Linear term
            for i in range(num_vertex):
                self.h_constraint[i, i] += alpha * (1 - num_vertex)
            self.const_constraint = alpha * num_vertex**2 / 4
