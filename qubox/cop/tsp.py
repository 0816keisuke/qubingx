import itertools

from typing import List
import numpy as np

from qubox.cop.base import Matrix, Model
from qubox.cop.qubo import QUBO


class TSP(QUBO):
    def __init__(
        self,
        distance_mtx: List[List[float]] | np.ndarray,
        ALPHA: float = 1,
        MODEL: str = "QUBO",
        MATRIX: str = "upper",
    ):
        NUM_CITY: int = len(distance_mtx)
        if isinstance(distance_mtx, np.ndarray):
            distance_mtx = distance_mtx.tolist()

        super().__init__(
            MODEL=Model(MODEL),
            MATRIX=Matrix(MATRIX),
            num_spin=NUM_CITY * NUM_CITY,
            q_all=np.zeros((NUM_CITY**2, NUM_CITY**2)),
            q_obj=np.zeros((NUM_CITY**2, NUM_CITY**2)),
            q_constraint=np.zeros((NUM_CITY**2, NUM_CITY**2)),
            const_all=0,
            const_obj=0,
            const_constraint=0,
        )

        self.h_obj(NUM_CITY, distance_mtx)
        self.h_constraint(NUM_CITY, ALPHA)
        self.h_all()

    # <<< Objective term >>>
    def h_obj(self, NUM_CITY: int, distance_mtx: List[List[float]]) -> None:
        for t_u_v in itertools.product(range(NUM_CITY), repeat=3):
            t, u, v = t_u_v[0], t_u_v[1], t_u_v[2]
            idx_i = NUM_CITY * t + u
            if t < NUM_CITY - 1:
                idx_j = NUM_CITY * t + v
            elif t == NUM_CITY - 1:
                idx_j = v
            self.q_obj[idx_i, idx_j] += distance_mtx[u][v]
        self.q_obj = (
            np.triu(self.q_obj) + np.tril(self.q_obj).T - np.diag(np.diag(self.q_obj))
        )

    # <<< Constraint term >>>
    def h_constraint(self, NUM_CITY, ALPHA) -> None:
        # Calculate constraint term1 : 1-hot of horizontal line
        # Quadratic term
        for t in range(NUM_CITY):
            for u in range(NUM_CITY - 1):
                for v in range(u + 1, NUM_CITY):
                    self.q_constraint[NUM_CITY * t + u, NUM_CITY * t + v] += ALPHA * 2
        # Linear term
        for t_u in itertools.product(range(NUM_CITY), repeat=2):
            self.q_constraint[
                NUM_CITY * t_u[0] + t_u[1], NUM_CITY * t_u[0] + t_u[1]
            ] += ALPHA * (-1)
        self.const_constraint = ALPHA * NUM_CITY

        # Calculate constraint term2 : 1-hot of vertical line
        # Quadratic term
        for u in range(NUM_CITY):
            for t1 in range(NUM_CITY - 1):
                for t2 in range(t1 + 1, NUM_CITY):
                    self.q_constraint[NUM_CITY * t1 + u, NUM_CITY * t2 + u] += ALPHA * 2
        # Linear term
        for u_t in itertools.product(range(NUM_CITY), repeat=2):
            self.q_constraint[
                NUM_CITY * u_t[1] + u_t[0], NUM_CITY * u_t[1] + u_t[0]
            ] += ALPHA * (-1)
        self.const_constraint += ALPHA * NUM_CITY
