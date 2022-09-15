import math
import numpy as np
from qubox.base import Base

class Knapsack(Base):
    def __init__(self,
                value_list,
                weight_list,
                max_weight,
                encoding="one-hot",
                ALPHA=1
                ):
        # Check tye type of Arguments
        if isinstance(weight_list, list):
            weight_list = np.array(weight_list)
        elif isinstance(weight_list, np.ndarray):
            pass
        else:
            print("The type of the argument 'weight_list' is WRONG.")
            print("It shoud be list/numpy.ndarray.")
            exit()
        if isinstance(value_list, list):
            value_list = np.array(value_list)
        elif isinstance(value_list, np.ndarray):
            pass
        else:
            print("The type of the argument 'value_list' is WRONG.")
            print("It shoud be list/numpy.ndarray.")
            exit()
        if not isinstance(max_weight, int):
            print("The type of the argument 'max_weight' is WRONG.")
            print("It shoud be int.")
            exit()
        if not encoding in ["one-hot", "log"]:
            print("The argument 'encoding' is WRONG.")
            print("It shoud be one-hot/log.")
            exit()

        NUM_ITEM = len(value_list)
        if encoding == "one-hot":
            super().__init__(num_spin = len(value_list) + max_weight)
        elif encoding == "log":
            super().__init__(num_spin = len(value_list) + math.floor(math.log(max_weight-1, 2)) + 1)
        np.set_printoptions(edgeitems=10) # Chenge the setting for printing numpy

        self.h_cost(NUM_ITEM, value_list)
        self.h_pen(encoding, NUM_ITEM, weight_list, max_weight, ALPHA)
        self.h_all()

    def h_cost(self, NUM_ITEM, value_list):
        # Cost term
        for a in range(NUM_ITEM):
            coef = -1 * value_list[a]
            self.q_cost[a, a] += coef

    def h_pen(self, encoding, NUM_ITEM, weight_list, max_weight, ALPHA):
        # 1-hot encoding
        # H = ( (\sum_(n=1)^W y_n) - 1 )^2 + ( \sum_(n=1)^W n y_n - \sum_(a=0)^(N-1) w_a x_a )^2
        if encoding == "one-hot":
            # 1-hot constraint
            #   ( (\sum_(n=1)^W y_n) - 1 )^2
            # = \sum_(n=1)^W-1 \sum_(m=n+1)^W 2 * y_n y_m - \sum_(n=1)^W y_n + 1
            # Quadratic term
            for n in range(1, max_weight):
                for m in range(n+1, max_weight+1):
                    coef = 2
                    idx_i = NUM_ITEM + n - 1
                    idx_j = NUM_ITEM + m - 1
                    self.q_pen[idx_i, idx_j] += ALPHA * coef
            # Linear term
            for n in range(1, max_weight+1):
                coef = -1
                idx = NUM_ITEM + n - 1
                self.q_pen[idx, idx] += ALPHA * coef
            # Constant term
            self.const_pen[0] += ALPHA

            # 1-hot encoding
            #   ( \sum_(n=1)^W n y_n - \sum_(a=0)^(N-1) w_a x_a )^2
            # = ( \sum_(n=1)^W n y_n )^2 - 2 * \sum_(n=1)^W n y_n * \sum_(a=0)^(N-1) w_a x_a + ( \sum_(a=0)^(N-1) w_a x_a )^2

            #   ( \sum_(n=1)^W n y_n )^2
            # = \sum_(n=1)^W-1 \sum_(m=n+1)^W 2 * n m y_n y_m + \sum_(n=1)^W n^2 y_n
            # Quadratic term
            for n in range(1, max_weight):
                for m in range(n+1, max_weight+1):
                    coef = 2 * n * m
                    idx_i = NUM_ITEM + n - 1
                    idx_j = NUM_ITEM + m - 1
                    self.q_pen[idx_i, idx_j] += ALPHA * coef
            # Linear term
            for n in range(1, max_weight+1):
                coef = n ** 2
                idx = NUM_ITEM + n - 1
                self.q_pen[idx, idx] += ALPHA * coef

            #   \sum_(n=1)^W n y_n * \sum_(a=0)^(N-1) w_a x_a
            # = \sum_(n=1)^W \sum_(a=0)^(N-1) n w_a y_n x_a
            # Quadratic term
            for n in range(1, max_weight+1):
                for a in range(NUM_ITEM):
                    coef = -2 * n * weight_list[a]
                    idx_i = NUM_ITEM + n - 1
                    idx_j = a
                    self.q_pen[idx_i, idx_j] += ALPHA * coef

            #   \sum_(a=0)^(N-1) w_a x_a
            # = \sum_(a=0)^(N-2) \sum_(b=a+1)^(N-1) 2 * w_a w_b x_a x_b + \sum_(a=0)^(N-1) (w_a)^2 x_a
            # Quadratic term
            for a in range(NUM_ITEM-1):
                for b in range(a+1, NUM_ITEM):
                    coef = 2 * weight_list[a] * weight_list[b]
                    idx_i = a
                    idx_j = b
                    self.q_pen[idx_i, idx_j] += ALPHA * coef
            # Linear term
            for a in range(NUM_ITEM):
                coef = weight_list[a] ** 2
                idx = a
                self.q_pen[idx, idx] += ALPHA * coef

        elif encoding == "log":
            #   { W -  ( \sum_(a=0)^(N-1) w_a x_a ) - ( \sum_(n=0)^([log_2(W-1)]) 2^n y_n ) }^2
            # = W^2 - 2 * W * ( \sum_(a=0)^(N-1) w_a x_a ) - 2 * W * ( \sum_(n=0)^([log_2(W-1)]) 2^n y_n )
            #       + ( \sum_(a=0)^(N-1) w_a x_a )^2 + 2 * ( \sum_(a=0)^(N-1) w_a x_a ) * ( \sum_(n=0)^([log_2(W-1)]) 2^n y_n )
            #       + ( \sum_(n=0)^([log_2(W-1)]) 2^n y_n )^2

            # [log_2(W-1)]
            num_binary = math.floor(math.log(max_weight-1, 2))

            # W^2
            self.const_pen[0] += ALPHA * max_weight * max_weight

            # -2 * W * (\sum_(a=0)^(N-1) w_a x_a)
            for a in range(NUM_ITEM):
                coef = -2 * max_weight * weight_list[a]
                idx = a
                self.q_pen[idx, idx] += ALPHA * coef

            # -2 * W * (\sum_(n=0)^([log_2(W-1)]) 2^n y_n)
            for n in range(num_binary+1):
                coef = -2 * max_weight * pow(2, n)
                idx = NUM_ITEM + n
                self.q_pen[idx, idx] += ALPHA * coef

            #   (\sum_(a=0)^(N-1) w_a x_a)^2
            # = \sum_(a=0)^(N-2) \sum_(b=a+1)^(N-1) 2 * w_a w_b x_a x_b + \sum_(a=0)^(N-1) (w_a)^2 x_a
            # Quadratic term
            for a in range(NUM_ITEM-1):
                for b in range(a+1, NUM_ITEM):
                    coef = 2 * weight_list[a] * weight_list[b]
                    idx_i = a
                    idx_j = b
                    self.q_pen[idx_i, idx_j] += ALPHA * coef
            # Linear term
            for a in range(NUM_ITEM):
                coef = weight_list[a] ** 2
                idx = a
                self.q_pen[idx, idx] += ALPHA * coef

            #   2 * (\sum_(a=0)^(N-1) w_a x_a) * (\sum_(n=0)^([log_2(W-1)]) 2^n y_n)
            # = \sum_(a=0)^(N-1) \sum_(n=0)^([log_2(W-1)]) w_a 2^(n+1) x_a y_n
            # Quadratic term
            for a in range(NUM_ITEM):
                for n in range(num_binary+1):
                    coef = weight_list[a] * pow(2, n+1)
                    idx_i = a
                    idx_j = NUM_ITEM + n
                    self.q_pen[idx_i, idx_j] += ALPHA * coef

            #   (\sum_(n=0)^([log_2(W-1)]) 2^n y_n)^2
            # = \sum_(n=0)^([log_2(W-1)]-1) \sum_(m=n+1)^([log_2(W-1)]) 2 * 2^n 2^m y_n y_m + \sum_(n=0)^([log_2(W-1)]) (2^n)^2 y_n
            # Quadratic term
            for n in range(num_binary):
                for m in range(n+1, num_binary+1):
                    coef = 2 * pow(2, n) * pow(2, m)
                    idx_i = NUM_ITEM + n
                    idx_j = NUM_ITEM + m
                    self.q_pen[idx_i, idx_j] += ALPHA * coef
            # Linear term
            for n in range(num_binary+1):
                coef = pow(2, n)**2
                idx = NUM_ITEM + n
                self.q_pen[idx, idx] += ALPHA * coef
