import itertools

from typing import List
import numpy as np

from qubingx.cop.base import Matrix, Model
from qubingx.cop.qubo import QUBO


class TSP(QUBO):
    def __init__(
        self,
        distance_mtx: List[List[float]] | np.ndarray,
        alpha: float = 1,
        MODEL: str = "QUBO",
        MATRIX: str = "upper",
    ):
        num_city: int = len(distance_mtx)
        if isinstance(distance_mtx, np.ndarray):
            distance_mtx = distance_mtx.tolist()

        super().__init__(
            MODEL=Model(MODEL),
            MATRIX=Matrix(MATRIX),
            num_spin=num_city * num_city,
            q_all=np.zeros((num_city**2, num_city**2)),
            q_obj=np.zeros((num_city**2, num_city**2)),
            q_constraint=np.zeros((num_city**2, num_city**2)),
            const_all=0,
            const_obj=0,
            const_constraint=0,
        )

        self.h_obj(num_city, distance_mtx)
        self.h_constraint(num_city, alpha)
        self.h_all()

    # <<< Objective term >>>
    def h_obj(self, num_city: int, distance_mtx: List[List[float]]) -> None:
        for t, u, v in itertools.product(range(num_city), repeat=3):
            idx_i = num_city * t + u
            if t < num_city - 1:
                idx_j = num_city * t + v
            elif t == num_city - 1:
                idx_j = v
            self.q_obj[idx_i, idx_j] += distance_mtx[u][v]
        self.q_obj = (
            np.triu(self.q_obj) + np.tril(self.q_obj).T - np.diag(np.diag(self.q_obj))
        )

    # <<< Constraint term >>>
    def h_constraint(self, num_city, alpha) -> None:
        # Calculate constraint term1 : 1-hot of horizontal line
        # Quadratic term
        for t in range(num_city):
            for u in range(num_city - 1):
                for v in range(u + 1, num_city):
                    self.q_constraint[num_city * t + u, num_city * t + v] += alpha * 2
        # Linear term
        for t, u in itertools.product(range(num_city), repeat=2):
            self.q_constraint[num_city * t + u, num_city * t + u] += alpha * (-1)
        self.const_constraint = alpha * num_city

        # Calculate constraint term2 : 1-hot of vertical line
        # Quadratic term
        for u in range(num_city):
            for t1 in range(num_city - 1):
                for t2 in range(t1 + 1, num_city):
                    self.q_constraint[num_city * t1 + u, num_city * t2 + u] += alpha * 2
        # Linear term
        for u, t in itertools.product(range(num_city), repeat=2):
            self.q_constraint[num_city * u + t, num_city * u + t] += alpha * (-1)
        self.const_constraint += alpha * num_city
