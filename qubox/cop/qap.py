import itertools
import numpy as np
from typing import List
from qubox.cop.base import Matrix, Model
from qubox.cop.qubo import QUBO


class QAP(QUBO):
    def __init__(
        self,
        distance_mtx: List[List[int | float]] | np.ndarray,
        weight_mtx: List[List[int | float]] | np.ndarray,
        ALPHA: float = 1,
        MODEL: str = "QUBO",
        MATRIX: str = "upper",
    ):
        if isinstance(weight_mtx, np.ndarray):
            weight_mtx = weight_mtx.tolist()
        if isinstance(distance_mtx, np.ndarray):
            distance_mtx = distance_mtx.tolist()

        NUM_FACTORY: int = len(weight_mtx)
        super().__init__(
            MODEL=Model(MODEL),
            MATRIX=Matrix(MATRIX),
            num_spin=NUM_FACTORY * NUM_FACTORY,
            q_all=np.zeros((NUM_FACTORY**2, NUM_FACTORY**2)),
            q_obj=np.zeros((NUM_FACTORY**2, NUM_FACTORY**2)),
            q_constraint=np.zeros((NUM_FACTORY**2, NUM_FACTORY**2)),
            const_all=0,
            const_obj=0,
            const_constraint=0,
        )
        self.h_obj(NUM_FACTORY, weight_mtx, distance_mtx)
        self.h_constraint(NUM_FACTORY, ALPHA)
        self.h_all()

    # <<< Objective term >>>
    def h_obj(
        self,
        NUM_FACTORY: int,
        weight_mtx: List[List[float]],
        distance_mtx: List[List[float]],
    ):
        for i_j_k_l in itertools.product(range(NUM_FACTORY), repeat=4):
            i, j, k, l = i_j_k_l[0], i_j_k_l[1], i_j_k_l[2], i_j_k_l[3]
            idx_i = NUM_FACTORY * i + k
            idx_j = NUM_FACTORY * j + l
            coef = distance_mtx[i][j] * weight_mtx[k][l]
            if coef == 0:
                continue
            self.q_obj[idx_i, idx_j] += coef
        self.q_obj = (
            np.triu(self.q_obj) + np.tril(self.q_obj).T - np.diag(np.diag(self.q_obj))
        )

    # <<< Constraint term >>>
    def h_constraint(self, NUM_FACTORY, ALPHA):
        # Calculate constraint term1 : 1-hot of horizontal line
        # Quadratic term
        for i in range(NUM_FACTORY):
            for k in range(NUM_FACTORY - 1):
                for l in range(k + 1, NUM_FACTORY):
                    self.q_constraint[NUM_FACTORY * i + k, NUM_FACTORY * i + l] += (
                        ALPHA * 2
                    )
        # Linear term
        for i_k in itertools.product(range(NUM_FACTORY), repeat=2):
            self.q_constraint[
                NUM_FACTORY * i_k[0] + i_k[1], NUM_FACTORY * i_k[0] + i_k[1]
            ] += ALPHA * (-1)
        self.const_constraint = ALPHA * NUM_FACTORY

        # Calculate constraint term2 : 1-hot of vertical line
        # Quadratic term
        for k in range(NUM_FACTORY):
            for i in range(NUM_FACTORY - 1):
                for j in range(i + 1, NUM_FACTORY):
                    self.q_constraint[NUM_FACTORY * i + k, NUM_FACTORY * j + k] += (
                        ALPHA * 2
                    )
        # Linear term
        for k_i in itertools.product(range(NUM_FACTORY), repeat=2):
            self.q_constraint[
                NUM_FACTORY * k_i[1] + k_i[0], NUM_FACTORY * k_i[1] + k_i[0]
            ] += ALPHA * (-1)
        self.const_constraint += ALPHA * NUM_FACTORY
