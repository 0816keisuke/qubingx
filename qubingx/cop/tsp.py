import itertools

import numpy as np
import numpy.typing as npt

from qubingx.cop.base import Base, Model


class TSP(Base):
    def __init__(
        self,
        distance_mtx: list[list[int | float]] | npt.NDArray,
        alpha: float = 1,
    ):
        num_city: int = len(distance_mtx)
        if isinstance(distance_mtx, np.ndarray):
            distance_mtx = distance_mtx.tolist()

        super().__init__(
            model=Model("QUBO"),
            num_spin=num_city * num_city,
            h_all=np.zeros((num_city**2, num_city**2)),
            h_obj=np.zeros((num_city**2, num_city**2)),
            h_constraint=np.zeros((num_city**2, num_city**2)),
            const_all=0,
            const_obj=0,
            const_constraint=0,
        )

        self._make_h_obj(num_city, distance_mtx)
        self._make_h_constraint(num_city, alpha)
        self._make_h_all()

    # <<< Objective term >>>
    def _make_h_obj(self, num_city: int, distance_mtx: list[list[float]]) -> None:
        for t, u, v in itertools.product(range(num_city), repeat=3):
            idx_i = num_city * t + u
            if t < num_city - 1:
                idx_j = num_city * t + v
            elif t == num_city - 1:
                idx_j = v
            self.h_obj[idx_i, idx_j] += distance_mtx[u][v]
        self.h_obj = np.triu(self.h_obj) + np.tril(self.h_obj).T - np.diag(np.diag(self.h_obj))

    # <<< Constraint term >>>
    def _make_h_constraint(self, num_city: int, alpha: float) -> None:
        # Calculate constraint term1 : 1-hot of horizontal line
        # Quadratic term
        for t in range(num_city):
            for u in range(num_city - 1):
                for v in range(u + 1, num_city):
                    self.h_constraint[num_city * t + u, num_city * t + v] += 2 * alpha
        # Linear term
        for t, u in itertools.product(range(num_city), repeat=2):
            self.h_constraint[num_city * t + u, num_city * t + u] += (-1) * alpha
        self.const_constraint = alpha * num_city

        # Calculate constraint term2 : 1-hot of vertical line
        # Quadratic term
        for u in range(num_city):
            for t1 in range(num_city - 1):
                for t2 in range(t1 + 1, num_city):
                    self.h_constraint[num_city * t1 + u, num_city * t2 + u] += 2 * alpha
        # Linear term
        for u, t in itertools.product(range(num_city), repeat=2):
            self.h_constraint[num_city * u + t, num_city * u + t] += (-1) * alpha
        self.const_constraint += alpha * num_city
