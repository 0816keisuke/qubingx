import math
from typing import Literal

import numpy as np
import numpy.typing as npt

from qubingx.cop.base import Base, Encoding, Model


class QKP(Base):
    def __init__(
        self,
        values: list[list[int]] | npt.NDArray,
        weights: list[int] | npt.NDArray,
        capacity: int,
        encoding: Literal["1-hot", "binary", "unary"] = "1-hot",
        alpha: float = 1.0,
    ):
        num_item: int = len(values)

        if isinstance(weights, np.ndarray):
            weights = weights.tolist()
        if isinstance(values, np.ndarray):
            values = values.tolist()

        if encoding == Encoding.one_hot.value:
            super().__init__(
                model=Model("QUBO"),
                num_spin=num_item + capacity,
                h_all=np.zeros((num_item + capacity, num_item + capacity)),
                h_obj=np.zeros((num_item + capacity, num_item + capacity)),
                h_constraint=np.zeros((num_item + capacity, num_item + capacity)),
                offset_all=0,
                offset_obj=0,
                offset_constraint=0,
            )
        elif encoding == Encoding.binary.value:
            num_binary = math.floor(math.log(capacity - 1, 2)) + 1
            num_spin = num_item + num_binary
            super().__init__(
                model=Model("QUBO"),
                num_spin=num_spin,
                h_all=np.zeros((num_spin, num_spin)),
                h_obj=np.zeros((num_spin, num_spin)),
                h_constraint=np.zeros((num_spin, num_spin)),
                offset_all=0,
                offset_obj=0,
                offset_constraint=0,
            )
        elif encoding == Encoding.unary.value:
            pass
        else:
            raise EncodingNameError(encoding)

        self._make_h_obj(num_item=num_item, values=values)
        self._make_h_constraint(
            encoding=encoding,
            num_item=num_item,
            weights=weights,
            capacity=capacity,
            alpha=alpha,
        )
        self._make_h_all()

    def _make_h_obj(self, num_item: int, values: list[list[int]]):
        # Cost term
        for i in range(num_item):
            for j in range(i, num_item):
                coef = -1 * values[i][j]
                self.h_obj[i, j] += coef

    def _make_h_constraint(
        self,
        encoding,
        num_item: int,
        weights: list[int] | np.ndarray,
        capacity: int,
        alpha: float = 1,
    ):
        # 1-hot encoding
        # H = ( (\sum_(n=1)^W y_n) - 1 )^2 + ( \sum_(n=1)^W n y_n - \sum_(a=0)^(N-1) w_a x_a )^2
        if encoding == Encoding.one_hot.value:
            # 1-hot constraint
            #   ( (\sum_(n=1)^W y_n) - 1 )^2
            # = \sum_(n=1)^W-1 \sum_(m=n+1)^W 2 * y_n y_m - \sum_(n=1)^W y_n + 1
            # Quadratic term
            for n in range(1, capacity):
                for m in range(n + 1, capacity + 1):
                    coef = 2
                    idx_i = num_item + n - 1
                    idx_j = num_item + m - 1
                    self.h_constraint[idx_i, idx_j] += alpha * coef
            # Linear term
            for n in range(1, capacity + 1):
                coef = -1
                idx = num_item + n - 1
                self.h_constraint[idx, idx] += alpha * coef
            # Constant term
            self.offset_constraint += alpha

            # 1-hot encoding
            #   ( \sum_(n=1)^W n y_n - \sum_(a=0)^(N-1) w_a x_a )^2
            # = ( \sum_(n=1)^W n y_n )^2 - 2 * \sum_(n=1)^W n y_n * \sum_(a=0)^(N-1) w_a x_a + ( \sum_(a=0)^(N-1) w_a x_a )^2

            #   ( \sum_(n=1)^W n y_n )^2
            # = \sum_(n=1)^W-1 \sum_(m=n+1)^W 2 * n m y_n y_m + \sum_(n=1)^W n^2 y_n
            # Quadratic term
            for n in range(1, capacity):
                for m in range(n + 1, capacity + 1):
                    coef = 2 * n * m
                    idx_i = num_item + n - 1
                    idx_j = num_item + m - 1
                    self.h_constraint[idx_i, idx_j] += alpha * coef
            # Linear term
            for n in range(1, capacity + 1):
                coef = n**2
                idx = num_item + n - 1
                self.h_constraint[idx, idx] += alpha * coef

            #   \sum_(n=1)^W n y_n * \sum_(a=0)^(N-1) w_a x_a
            # = \sum_(n=1)^W \sum_(a=0)^(N-1) n w_a y_n x_a
            # Quadratic term
            for n in range(1, capacity + 1):
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

        elif encoding == Encoding.binary.value:
            #   { W -  ( \sum_(a=0)^(N-1) w_a x_a ) - ( \sum_(n=0)^([log_2(W-1)]) 2^n y_n ) }^2
            # = W^2 - 2 * W * ( \sum_(a=0)^(N-1) w_a x_a ) - 2 * W * ( \sum_(n=0)^([log_2(W-1)]) 2^n y_n )
            #       + ( \sum_(a=0)^(N-1) w_a x_a )^2 + 2 * ( \sum_(a=0)^(N-1) w_a x_a ) * ( \sum_(n=0)^([log_2(W-1)]) 2^n y_n )
            #       + ( \sum_(n=0)^([log_2(W-1)]) 2^n y_n )^2

            # [log_2(W-1)]
            num_binary = math.floor(math.log(capacity - 1, 2))

            # W^2
            self.offset_constraint += alpha * capacity * capacity

            # -2 * W * (\sum_(a=0)^(N-1) w_a x_a)
            for a in range(num_item):
                coef = -2 * capacity * weights[a]
                idx = a
                self.h_constraint[idx, idx] += alpha * coef

            # -2 * W * (\sum_(n=0)^([log_2(W-1)]) 2^n y_n)
            for n in range(num_binary + 1):
                coef = -2 * capacity * pow(2, n)
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

        elif encoding == Encoding.unary.value:
            pass
