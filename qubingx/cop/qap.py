import itertools
import numpy as np
from typing import List
from qubingx.cop.base import Matrix, Model
from qubingx.cop.qubo import QUBO


class QAP(QUBO):
    def __init__(
        self,
        weight_mtx: List[List[int | float]] | np.ndarray,
        distance_mtx: List[List[int | float]] | np.ndarray,
        alpha: float = 1,
        MODEL: str = "QUBO",
        MATRIX: str = "upper",
    ):
        if isinstance(weight_mtx, np.ndarray):
            weight_mtx = weight_mtx.tolist()
        if isinstance(distance_mtx, np.ndarray):
            distance_mtx = distance_mtx.tolist()

        num_facility: int = len(weight_mtx)
        super().__init__(
            MODEL=Model(MODEL),
            MATRIX=Matrix(MATRIX),
            num_spin=num_facility * num_facility,
            q_all=np.zeros((num_facility**2, num_facility**2)),
            q_obj=np.zeros((num_facility**2, num_facility**2)),
            q_constraint=np.zeros((num_facility**2, num_facility**2)),
            const_all=0.0,
            const_obj=0.0,
            const_constraint=0,
        )
        self.h_obj(num_facility, weight_mtx, distance_mtx)
        self.h_constraint(num_facility, alpha)
        self.h_all()

    # <<< Objective term >>>
    def h_obj(
        self,
        num_facility: int,
        weight_mtx: List[List[float]],
        distance_mtx: List[List[float]],
    ):
        for i, j, k, l in itertools.product(range(num_facility), repeat=4):
            idx_i = num_facility * i + k
            idx_j = num_facility * j + l
            coef = distance_mtx[i][j] * weight_mtx[k][l]
            if coef == 0:
                continue
            self.q_obj[idx_i, idx_j] += coef
        self.q_obj = (
            np.triu(self.q_obj) + np.tril(self.q_obj).T - np.diag(np.diag(self.q_obj))
        )

    # <<< Constraint term >>>
    def h_constraint(self, num_facility: int, alpha: int = 1):
        # Calculate constraint term1 : 1-hot of horizontal line
        # Quadratic term
        for i in range(num_facility):
            for k in range(num_facility - 1):
                for l in range(k + 1, num_facility):
                    self.q_constraint[num_facility * i + k, num_facility * i + l] += (
                        alpha * 2
                    )
        # Linear term
        for i, k in itertools.product(range(num_facility), repeat=2):
            self.q_constraint[num_facility * i + k, num_facility * i + k] += alpha * (
                -1
            )
        self.const_constraint = alpha * num_facility

        # Calculate constraint term2 : 1-hot of vertical line
        # Quadratic term
        for k in range(num_facility):
            for i in range(num_facility - 1):
                for j in range(i + 1, num_facility):
                    self.q_constraint[num_facility * i + k, num_facility * j + k] += (
                        alpha * 2
                    )
        # Linear term
        for k, i in itertools.product(range(num_facility), repeat=2):
            self.q_constraint[num_facility * k + i, num_facility * k + i] += alpha * (
                -1
            )
        self.const_constraint += alpha * num_facility
