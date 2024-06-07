import math

import numpy as np

from qubingx.cop.base import Base, Encoding, Model


class MDKP(Base):
    def __init__(
        self,
        value_list: list[int | float] | np.ndarray,
        weight_list: list[list[int | float]] | np.ndarray,
        max_weight_list: list[int | float] | np.ndarray,
        dim,
        encoding: str = "1-hot",
        alpha: float = 1,
    ):
        self.encoding = Encoding(encoding)
        num_item = len(value_list)

        if isinstance(value_list, np.ndarray):
            value_list = value_list.to_list()
        if isinstance(weight_list, np.ndarray):
            weight_list = weight_list.to_list()
        if isinstance(max_weight_list, np.ndarray):
            max_weight_list = max_weight_list.to_list()

        if encoding == "1-hot":
            num_spin = num_item + sum(max_weight_list)
            super().__init__(
                model=Model("QUBO"),
                num_spin=num_item + sum(max_weight_list),
                q_all=np.zeros((num_spin, num_spin)),
                q_obj=np.zeros((num_spin, num_spin)),
                q_constraint=np.zeros((num_spin, num_spin)),
                const_all=0,
                const_obj=0,
                const_constraint=0,
            )
        elif encoding == "binary":
            num_slack_list = [
                0 if dim_i == -1 else math.floor(math.log(max_weight_list[dim_i] - 1, 2)) + 1
                for dim_i in range(-1, dim)
            ]
            num_spin = len(value_list) + sum(num_slack_list)
            super().__init__(
                model=Model("QUBO"),
                num_spin=num_spin,
                q_all=np.zeros((num_spin, num_spin)),
                q_obj=np.zeros((num_spin, num_spin)),
                q_constraint=np.zeros((num_spin, num_spin)),
                const_all=0,
                const_obj=0,
                const_constraint=0,
            )

        self.h_obj(num_item, value_list)
        self.h_constraint(
            encoding, dim, num_item, weight_list, max_weight_list, num_slack_list, alpha
        )
        self.h_all()

    def h_obj(self, num_item, value_list):
        # Cost term
        for a in range(num_item):
            coef = -1 * value_list[a]
            self.q_obj[a, a] += coef

    def h_constraint(
        self,
        encoding,
        dim,
        num_item,
        weight_list,
        max_weight_list,
        num_slack_list,
        alpha,
    ):
        if encoding == "1-hot":
            # At arbitrarily i-th dimension
            # H = \sum_d^Dim [ { \sum_(n=1)^(W_d) y_(d,n) - 1 }^2 + { \sum_(n=1)^(W_d) n y_(d,n) - \sum_(a=0)^(N-1) w_(d,a) x_a }^2 ]
            pass
        elif encoding == "log":
            # H = \sum_d^Dim { W_d -  ( \sum_(a=0)^(N-1) w_(d,a) x_a ) - (\sum_(n=0)^([log_2((W_d)-1)]) 2^n y_(d,n)) }^2
            #   = (W_d)^2 - 2 * W_d * ( \sum_(a=0)^(N-1) w_(d,a) x_a ) - 2 * W_d * ( \sum_(n=0)^([log_2((W_d)-1)]) 2^n y_(d,n) )
            #             + ( \sum_(a=0)^(N-1) w_(d,a) x_a )^2 + 2 * ( \sum_(a=0)^(N-1) w_(d,a) x_a ) * ( \sum_(n=0)^([log_2((W_d)-1)]) 2^n y_(d,n) )
            #             + ( \sum_(n=0)^([log_2((W_d)-1)]) 2^n y_(d,n) )^2
            for dim_i in range(dim):
                # [log_2((W_d)-1)]
                num_binary = math.floor(math.log(max_weight_list[dim_i] - 1, 2))

                # (W_d)^2
                self.const_pen += alpha * max_weight_list[dim_i] * max_weight_list[dim_i]

                # -2 * W_d * ( \sum_(a=0)^(N-1) w_(d,a) x_a )
                for a in range(num_item):
                    coef = -2 * max_weight_list[dim_i] * weight_list[dim_i, a]
                    idx = a
                    self.q_constraint[idx, idx] += alpha * coef

                # -2 * W_d * ( \sum_(n=0)^([log_2((W_d)-1)]) 2^n y_(d,n) )
                for n in range(num_binary + 1):
                    coef = -2 * max_weight_list[dim_i] * pow(2, n)
                    idx = num_item + sum(num_slack_list[: dim_i + 1]) + n
                    self.q_constraint[idx, idx] += alpha * coef

                #   ( \sum_(a=0)^(N-1) w_(d,a) x_a )^2
                # = \sum_(a=0)^(N-2) \sum_(b=a+1)^(N-1) 2 * w_(d,a) w_(d,b) x_a x_b + \sum_(a=0)^(N-1) (w_(d,a))^2 x_a
                # Quadratic term
                for a in range(num_item - 1):
                    for b in range(a + 1, num_item):
                        coef = 2 * weight_list[dim_i, a] * weight_list[dim_i, b]
                        idx_i = a
                        idx_j = b
                        self.q_constraint[idx_i, idx_j] += alpha * coef
                # Linear term
                for a in range(num_item):
                    coef = weight_list[dim_i, a] ** 2
                    idx = a
                    self.q_constraint[idx, idx] += alpha * coef

                #   2 * ( \sum_(a=0)^(N-1) w_(d,a) x_a) * ( \sum_(n=0)^([log_2((W_d)-1)]) 2^n y_(d,n) )
                # = \sum_(a=0)^(N-1) \sum_(n=0)^([log_2((W_d)-1)]) w_(d,a) 2^(n+1) x_a y_(d,n)
                # Quadratic term
                for a in range(num_item):
                    for n in range(num_binary + 1):
                        coef = weight_list[dim_i, a] * pow(2, n + 1)
                        idx_i = a
                        idx_j = num_item + sum(num_slack_list[: dim_i + 1]) + n
                        self.q_constraint[idx_i, idx_j] += alpha * coef

                #   ( \sum_(n=0)^([log_2((W_d)-1)]) 2^n y_(d,n) )^2
                # = \sum_(n=0)^([log_2((W_d)-1)]-1) \sum_(m=n+1)^([log_2((W_d)-1)]) 2 * 2^n 2^m y_(d,n) y_(d,m) + \sum_(n=0)^([log_2((W_d)-1)]) (2^n)^2 y_(d,n)
                # Quadratic term
                for n in range(num_binary):
                    for m in range(n + 1, num_binary + 1):
                        coef = 2 * pow(2, n) * pow(2, m)
                        idx_i = num_item + sum(num_slack_list[: dim_i + 1]) + n
                        idx_j = num_item + sum(num_slack_list[: dim_i + 1]) + m
                        self.q_constraint[idx_i, idx_j] += alpha * coef
                # Linear term
                for n in range(num_binary + 1):
                    coef = pow(2, n) ** 2
                    idx = num_item + sum(num_slack_list[: dim_i + 1]) + n
                    self.q_constraint[idx, idx] += alpha * coef
