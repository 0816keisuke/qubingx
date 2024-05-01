import itertools

import numpy as np
import numpy.typing as npt

from qubingx.cop.base import Base, Model


class QAP(Base):
    def __init__(
        self,
        weight_mtx: list[list[int | float]] | npt.NDArray,
        distance_mtx: list[list[int | float]] | npt.NDArray,
        alpha: float = 1.0,
    ):
        if isinstance(weight_mtx, np.ndarray):
            weight_mtx = weight_mtx.tolist()
        if isinstance(distance_mtx, np.ndarray):
            distance_mtx = distance_mtx.tolist()

        num_facility: int = len(weight_mtx)
        super().__init__(
            model=Model("QUBO"),
            num_spin=num_facility * num_facility,
            h_all=np.zeros((num_facility**2, num_facility**2)),
            h_obj=np.zeros((num_facility**2, num_facility**2)),
            h_constraint=np.zeros((num_facility**2, num_facility**2)),
            const_all=0.0,
            const_obj=0.0,
            const_constraint=0,
        )
        self._make_h_obj(num_facility, weight_mtx, distance_mtx)
        self._make_h_constraint(num_facility, alpha)
        self._make_h_all()

    # <<< Objective term >>>
    def _make_h_obj(
        self,
        num_facility: int,
        weight_mtx: list[list[int | float]],
        distance_mtx: list[list[int | float]],
    ):
        # TODO: クロネッカー積を使って書き直す
        for i, j, k, l in itertools.product(range(num_facility), repeat=4):
            idx_i = num_facility * i + k
            idx_j = num_facility * j + l
            coef = distance_mtx[i][j] * weight_mtx[k][l]
            if coef == 0:
                continue
            self.h_obj[idx_i, idx_j] += coef
        self.h_obj = np.triu(self.h_obj) + np.tril(self.h_obj).T - np.diag(np.diag(self.h_obj))

    # <<< Constraint term >>>
    def _make_h_constraint(self, num_facility: int, alpha: int = 1):
        # Calculate constraint term1 : 1-hot of horizontal line
        # Quadratic term
        for i in range(num_facility):
            for k in range(num_facility - 1):
                for l in range(k + 1, num_facility):
                    self.h_constraint[num_facility * i + k, num_facility * i + l] += 2 * alpha
        # Linear term
        for i, k in itertools.product(range(num_facility), repeat=2):
            self.h_constraint[num_facility * i + k, num_facility * i + k] += (-1) * alpha
        self.const_constraint = alpha * num_facility

        # Calculate constraint term2 : 1-hot of vertical line
        # Quadratic term
        for k in range(num_facility):
            for i in range(num_facility - 1):
                for j in range(i + 1, num_facility):
                    self.h_constraint[num_facility * i + k, num_facility * j + k] += 2 * alpha
        # Linear term
        for k, i in itertools.product(range(num_facility), repeat=2):
            self.h_constraint[num_facility * k + i, num_facility * k + i] += (-1) * alpha
        self.const_constraint += alpha * num_facility
