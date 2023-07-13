import math

import numpy as np
from typing import List

from qubox.cop.base import Matrix, Model, Encoding
from qubox.cop.qubo import QUBO


class Knapsack(QUBO):
    def __init__(
        self,
        value_list: List[int | float] | np.ndarray,
        weight_list: List[int | float] | np.ndarray,
        max_weight: float,
        encoding: str = "1-hot",
        ALPHA: float = 1,
        MODEL: str = "QUBO",
        MATRIX: str = "upper",
    ):
        self.encoding = Encoding(encoding)
        NUM_ITEM = len(value_list)

        if isinstance(weight_list, np.ndarray):
            weight_list = weight_list.tolist()
        if isinstance(value_list, np.ndarray):
            value_list = value_list.tolist()

        if encoding == "1-hot":
            super().__init__(
                MODEL=Model(MODEL),
                MATRIX=Matrix(MATRIX),
                num_spin=NUM_ITEM + max_weight,
                q_all=np.zeros((NUM_ITEM + max_weight, NUM_ITEM + max_weight)),
                q_obj=np.zeros((NUM_ITEM + max_weight, NUM_ITEM + max_weight)),
                q_constraint=np.zeros((NUM_ITEM + max_weight, NUM_ITEM + max_weight)),
                const_all=0,
                const_obj=0,
                const_constraint=0,
            )
        elif encoding == "binary":
            super().__init__(
                MODEL=Model(MODEL),
                MATRIX=Matrix(MATRIX),
                num_spin=NUM_ITEM + math.floor(math.log(max_weight - 1, 2)) + 1,
                q_all=np.zeros(
                    (
                        NUM_ITEM + math.floor(math.log(max_weight - 1, 2)) + 1,
                        NUM_ITEM + math.floor(math.log(max_weight - 1, 2)) + 1,
                    )
                ),
                q_obj=np.zeros(
                    (
                        NUM_ITEM + math.floor(math.log(max_weight - 1, 2)) + 1,
                        NUM_ITEM + math.floor(math.log(max_weight - 1, 2)) + 1,
                    )
                ),
                q_constraint=np.zeros(
                    (
                        NUM_ITEM + math.floor(math.log(max_weight - 1, 2)) + 1,
                        NUM_ITEM + math.floor(math.log(max_weight - 1, 2)) + 1,
                    )
                ),
                const_all=0,
                const_obj=0,
                const_constraint=0,
            )

        self.h_obj(NUM_ITEM=NUM_ITEM, value_list=value_list)
        self.h_constraint(
            encoding=encoding,
            NUM_ITEM=NUM_ITEM,
            weight_list=weight_list,
            max_weight=max_weight,
            ALPHA=ALPHA,
        )
        self.h_all()

    def h_obj(self, NUM_ITEM: int, value_list: List[float]):
        # Cost term
        for a in range(NUM_ITEM):
            coef = -1 * value_list[a]
            self.q_obj[a, a] += coef

    def h_constraint(
        self,
        encoding,
        NUM_ITEM: int,
        weight_list: List[float],
        max_weight: float,
        ALPHA: float = 1,
    ):
        # 1-hot encoding
        # H = ( (\sum_(n=1)^W y_n) - 1 )^2 + ( \sum_(n=1)^W n y_n - \sum_(a=0)^(N-1) w_a x_a )^2
        if encoding == "1-hot":
            # 1-hot constraint
            #   ( (\sum_(n=1)^W y_n) - 1 )^2
            # = \sum_(n=1)^W-1 \sum_(m=n+1)^W 2 * y_n y_m - \sum_(n=1)^W y_n + 1
            # Quadratic term
            for n in range(1, max_weight):
                for m in range(n + 1, max_weight + 1):
                    coef = 2
                    idx_i = NUM_ITEM + n - 1
                    idx_j = NUM_ITEM + m - 1
                    self.q_constraint[idx_i, idx_j] += ALPHA * coef
            # Linear term
            for n in range(1, max_weight + 1):
                coef = -1
                idx = NUM_ITEM + n - 1
                self.q_constraint[idx, idx] += ALPHA * coef
            # Constant term
            self.const_constraint += ALPHA

            # 1-hot encoding
            #   ( \sum_(n=1)^W n y_n - \sum_(a=0)^(N-1) w_a x_a )^2
            # = ( \sum_(n=1)^W n y_n )^2 - 2 * \sum_(n=1)^W n y_n * \sum_(a=0)^(N-1) w_a x_a + ( \sum_(a=0)^(N-1) w_a x_a )^2

            #   ( \sum_(n=1)^W n y_n )^2
            # = \sum_(n=1)^W-1 \sum_(m=n+1)^W 2 * n m y_n y_m + \sum_(n=1)^W n^2 y_n
            # Quadratic term
            for n in range(1, max_weight):
                for m in range(n + 1, max_weight + 1):
                    coef = 2 * n * m
                    idx_i = NUM_ITEM + n - 1
                    idx_j = NUM_ITEM + m - 1
                    self.q_constraint[idx_i, idx_j] += ALPHA * coef
            # Linear term
            for n in range(1, max_weight + 1):
                coef = n**2
                idx = NUM_ITEM + n - 1
                self.q_constraint[idx, idx] += ALPHA * coef

            #   \sum_(n=1)^W n y_n * \sum_(a=0)^(N-1) w_a x_a
            # = \sum_(n=1)^W \sum_(a=0)^(N-1) n w_a y_n x_a
            # Quadratic term
            for n in range(1, max_weight + 1):
                for a in range(NUM_ITEM):
                    coef = -2 * n * weight_list[a]
                    idx_i = NUM_ITEM + n - 1
                    idx_j = a
                    self.q_constraint[idx_i, idx_j] += ALPHA * coef

            #   \sum_(a=0)^(N-1) w_a x_a
            # = \sum_(a=0)^(N-2) \sum_(b=a+1)^(N-1) 2 * w_a w_b x_a x_b + \sum_(a=0)^(N-1) (w_a)^2 x_a
            # Quadratic term
            for a in range(NUM_ITEM - 1):
                for b in range(a + 1, NUM_ITEM):
                    coef = 2 * weight_list[a] * weight_list[b]
                    idx_i = a
                    idx_j = b
                    self.q_constraint[idx_i, idx_j] += ALPHA * coef
            # Linear term
            for a in range(NUM_ITEM):
                coef = weight_list[a] ** 2
                idx = a
                self.q_constraint[idx, idx] += ALPHA * coef

        elif encoding == "binary":
            #   { W -  ( \sum_(a=0)^(N-1) w_a x_a ) - ( \sum_(n=0)^([log_2(W-1)]) 2^n y_n ) }^2
            # = W^2 - 2 * W * ( \sum_(a=0)^(N-1) w_a x_a ) - 2 * W * ( \sum_(n=0)^([log_2(W-1)]) 2^n y_n )
            #       + ( \sum_(a=0)^(N-1) w_a x_a )^2 + 2 * ( \sum_(a=0)^(N-1) w_a x_a ) * ( \sum_(n=0)^([log_2(W-1)]) 2^n y_n )
            #       + ( \sum_(n=0)^([log_2(W-1)]) 2^n y_n )^2

            # [log_2(W-1)]
            num_binary = math.floor(math.log(max_weight - 1, 2))

            # W^2
            self.const_constraint += ALPHA * max_weight * max_weight

            # -2 * W * (\sum_(a=0)^(N-1) w_a x_a)
            for a in range(NUM_ITEM):
                coef = -2 * max_weight * weight_list[a]
                idx = a
                self.q_constraint[idx, idx] += ALPHA * coef

            # -2 * W * (\sum_(n=0)^([log_2(W-1)]) 2^n y_n)
            for n in range(num_binary + 1):
                coef = -2 * max_weight * pow(2, n)
                idx = NUM_ITEM + n
                self.q_constraint[idx, idx] += ALPHA * coef

            #   (\sum_(a=0)^(N-1) w_a x_a)^2
            # = \sum_(a=0)^(N-2) \sum_(b=a+1)^(N-1) 2 * w_a w_b x_a x_b + \sum_(a=0)^(N-1) (w_a)^2 x_a
            # Quadratic term
            for a in range(NUM_ITEM - 1):
                for b in range(a + 1, NUM_ITEM):
                    coef = 2 * weight_list[a] * weight_list[b]
                    idx_i = a
                    idx_j = b
                    self.q_constraint[idx_i, idx_j] += ALPHA * coef
            # Linear term
            for a in range(NUM_ITEM):
                coef = weight_list[a] ** 2
                idx = a
                self.q_constraint[idx, idx] += ALPHA * coef

            #   2 * (\sum_(a=0)^(N-1) w_a x_a) * (\sum_(n=0)^([log_2(W-1)]) 2^n y_n)
            # = \sum_(a=0)^(N-1) \sum_(n=0)^([log_2(W-1)]) w_a 2^(n+1) x_a y_n
            # Quadratic term
            for a in range(NUM_ITEM):
                for n in range(num_binary + 1):
                    coef = weight_list[a] * pow(2, n + 1)
                    idx_i = a
                    idx_j = NUM_ITEM + n
                    self.q_constraint[idx_i, idx_j] += ALPHA * coef

            #   (\sum_(n=0)^([log_2(W-1)]) 2^n y_n)^2
            # = \sum_(n=0)^([log_2(W-1)]-1) \sum_(m=n+1)^([log_2(W-1)]) 2 * 2^n 2^m y_n y_m + \sum_(n=0)^([log_2(W-1)]) (2^n)^2 y_n
            # Quadratic term
            for n in range(num_binary):
                for m in range(n + 1, num_binary + 1):
                    coef = 2 * pow(2, n) * pow(2, m)
                    idx_i = NUM_ITEM + n
                    idx_j = NUM_ITEM + m
                    self.q_constraint[idx_i, idx_j] += ALPHA * coef
            # Linear term
            for n in range(num_binary + 1):
                coef = pow(2, n) ** 2
                idx = NUM_ITEM + n
                self.q_constraint[idx, idx] += ALPHA * coef
