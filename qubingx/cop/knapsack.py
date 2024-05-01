import math
from typing import Literal

import numpy as np

from qubingx.cop.base import Base, Encoding, Model

ENCODING_TYPE = Literal["1-hot", "binary", "unary"]


class Knapsack(Base):
    def __init__(
        self,
        values: list[int] | np.ndarray,
        weights: list[int] | np.ndarray,
        max_weight: int,
        encoding: ENCODING_TYPE = "1-hot",
        alpha: float = 1.0,
    ):
        self.encoding = Encoding(encoding)
        num_item = len(values)

        if isinstance(weights, np.ndarray):
            weights = weights.tolist()
        if isinstance(values, np.ndarray):
            values = values.tolist()

        if encoding == "1-hot":
            num_spin = num_item + max_weight
            super().__init__(
                model=Model("QUBO"),
                num_spin=num_spin,
                h_all=np.zeros((num_spin, num_spin)),
                h_obj=np.zeros((num_spin, num_spin)),
                h_constraint=np.zeros((num_spin, num_spin)),
                const_all=0,
                const_obj=0,
                const_constraint=0,
            )
        elif encoding == "binary":
            num_spin = num_item + math.floor(math.log(max_weight - 1, 2)) + 1
            super().__init__(
                model=Model("QUBO"),
                num_spin=num_spin,
                h_all=np.zeros((num_spin, num_spin)),
                h_obj=np.zeros((num_spin, num_spin)),
                h_constraint=np.zeros((num_spin, num_spin)),
                const_all=0,
                const_obj=0,
                const_constraint=0,
            )

        self._make_h_obj(num_item=num_item, values=values)
        self._make_h_constraint(
            encoding=encoding,
            num_item=num_item,
            weights=weights,
            max_weight=max_weight,
            alpha=alpha,
        )
        self._make_h_all()

    def _make_h_obj(self, num_item: int, values: list[float]):
        # Cost term
        for a in range(num_item):
            coef = -1 * values[a]
            self.h_obj[a, a] += coef

    def _make_h_constraint(
        self,
        encoding: ENCODING_TYPE,
        num_item: int,
        weights: list[float],
        max_weight: float,
        alpha: float = 1.0,
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
                    idx_i = num_item + n - 1
                    idx_j = num_item + m - 1
                    self.h_constraint[idx_i, idx_j] += alpha * coef
            # Linear term
            for n in range(1, max_weight + 1):
                coef = -1
                idx = num_item + n - 1
                self.h_constraint[idx, idx] += alpha * coef
            # Constant term
            self.const_constraint += alpha

            # 1-hot encoding
            #   ( \sum_(n=1)^W n y_n - \sum_(a=0)^(N-1) w_a x_a )^2
            # = ( \sum_(n=1)^W n y_n )^2 - 2 * \sum_(n=1)^W n y_n * \sum_(a=0)^(N-1) w_a x_a + ( \sum_(a=0)^(N-1) w_a x_a )^2

            #   ( \sum_(n=1)^W n y_n )^2
            # = \sum_(n=1)^W-1 \sum_(m=n+1)^W 2 * n m y_n y_m + \sum_(n=1)^W n^2 y_n
            # Quadratic term
            for n in range(1, max_weight):
                for m in range(n + 1, max_weight + 1):
                    coef = 2 * n * m
                    idx_i = num_item + n - 1
                    idx_j = num_item + m - 1
                    self.h_constraint[idx_i, idx_j] += alpha * coef
            # Linear term
            for n in range(1, max_weight + 1):
                coef = n**2
                idx = num_item + n - 1
                self.h_constraint[idx, idx] += alpha * coef

            #   \sum_(n=1)^W n y_n * \sum_(a=0)^(N-1) w_a x_a
            # = \sum_(n=1)^W \sum_(a=0)^(N-1) n w_a y_n x_a
            # Quadratic term
            for n in range(1, max_weight + 1):
                for a in range(num_item):
                    coef = -2 * n * weights[a]
                    idx_i = num_item + n - 1
                    idx_j = a
                    self.h_constraint[idx_i, idx_j] += alpha * coef

            #   \sum_(a=0)^(N-1) w_a x_a
            # = \sum_(a=0)^(N-2) \sum_(b=a+1)^(N-1) 2 * w_a w_b x_a x_b + \sum_(a=0)^(N-1) (w_a)^2 x_a
            # Quadratic term
            for a in range(num_item - 1):
                for b in range(a + 1, num_item):
                    coef = 2 * weights[a] * weights[b]
                    idx_i = a
                    idx_j = b
                    self.h_constraint[idx_i, idx_j] += alpha * coef
            # Linear term
            for a in range(num_item):
                coef = weights[a] ** 2
                idx = a
                self.h_constraint[idx, idx] += alpha * coef

        elif encoding == "binary":
            #   { W -  ( \sum_(a=0)^(N-1) w_a x_a ) - ( \sum_(n=0)^([log_2(W-1)]) 2^n y_n ) }^2
            # = W^2 - 2 * W * ( \sum_(a=0)^(N-1) w_a x_a ) - 2 * W * ( \sum_(n=0)^([log_2(W-1)]) 2^n y_n )
            #       + ( \sum_(a=0)^(N-1) w_a x_a )^2 + 2 * ( \sum_(a=0)^(N-1) w_a x_a ) * ( \sum_(n=0)^([log_2(W-1)]) 2^n y_n )
            #       + ( \sum_(n=0)^([log_2(W-1)]) 2^n y_n )^2

            # [log_2(W-1)]
            num_binary = math.floor(math.log(max_weight - 1, 2))

            # W^2
            self.const_constraint += alpha * max_weight * max_weight

            # -2 * W * (\sum_(a=0)^(N-1) w_a x_a)
            for a in range(num_item):
                coef = -2 * max_weight * weights[a]
                idx = a
                self.h_constraint[idx, idx] += alpha * coef

            # -2 * W * (\sum_(n=0)^([log_2(W-1)]) 2^n y_n)
            for n in range(num_binary + 1):
                coef = -2 * max_weight * pow(2, n)
                idx = num_item + n
                self.h_constraint[idx, idx] += alpha * coef

            #   (\sum_(a=0)^(N-1) w_a x_a)^2
            # = \sum_(a=0)^(N-2) \sum_(b=a+1)^(N-1) 2 * w_a w_b x_a x_b + \sum_(a=0)^(N-1) (w_a)^2 x_a
            # Quadratic term
            for a in range(num_item - 1):
                for b in range(a + 1, num_item):
                    coef = 2 * weights[a] * weights[b]
                    idx_i = a
                    idx_j = b
                    self.h_constraint[idx_i, idx_j] += alpha * coef
            # Linear term
            for a in range(num_item):
                coef = weights[a] ** 2
                idx = a
                self.h_constraint[idx, idx] += alpha * coef

            #   2 * (\sum_(a=0)^(N-1) w_a x_a) * (\sum_(n=0)^([log_2(W-1)]) 2^n y_n)
            # = \sum_(a=0)^(N-1) \sum_(n=0)^([log_2(W-1)]) w_a 2^(n+1) x_a y_n
            # Quadratic term
            for a in range(num_item):
                for n in range(num_binary + 1):
                    coef = weights[a] * pow(2, n + 1)
                    idx_i = a
                    idx_j = num_item + n
                    self.h_constraint[idx_i, idx_j] += alpha * coef

            #   (\sum_(n=0)^([log_2(W-1)]) 2^n y_n)^2
            # = \sum_(n=0)^([log_2(W-1)]-1) \sum_(m=n+1)^([log_2(W-1)]) 2 * 2^n 2^m y_n y_m + \sum_(n=0)^([log_2(W-1)]) (2^n)^2 y_n
            # Quadratic term
            for n in range(num_binary):
                for m in range(n + 1, num_binary + 1):
                    coef = 2 * pow(2, n) * pow(2, m)
                    idx_i = num_item + n
                    idx_j = num_item + m
                    self.h_constraint[idx_i, idx_j] += alpha * coef
            # Linear term
            for n in range(num_binary + 1):
                coef = pow(2, n) ** 2
                idx = num_item + n
                self.h_constraint[idx, idx] += alpha * coef
