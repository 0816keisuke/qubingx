import math
import numpy as np
from .base import Base

class MKP(Base):
    def __init__(self,
                value_list,
                weight_list,
                max_weight_list,
                dim,
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
        if isinstance(max_weight_list, list):
            max_weight_list = np.array(max_weight_list)
        elif isinstance(max_weight_list, np.ndarray):
            pass
        else:
            print("The type of the argument 'max_weight_list' is WRONG.")
            print("It shoud be list/numpy.ndarray.")
            exit()
        if not encoding in ["one-hot", "log"]:
            print("The argument 'encoding' is WRONG.")
            print("It shoud be one-hot/log.")
            exit()

        NUM_ITEM = len(value_list)
        if encoding == "one-hot":
            super().__init__(NUM_SPIN = len(value_list) + max_weight)
        elif encoding == "log":
            num_slack_list = [0 if dim_i==-1 else math.floor(math.log(max_weight_list[dim_i]-1, 2)) + 1 for dim_i in range(-1, dim)]
            NUM_SPIN = len(value_list) + sum(num_slack_list)
            super().__init__(NUM_SPIN = NUM_SPIN)
        np.set_printoptions(edgeitems=10) # Chenge the setting for printing numpy

        self.cost_term(NUM_ITEM, value_list)
        self.penalty_term(encoding, dim, NUM_ITEM, weight_list, max_weight_list, num_slack_list, ALPHA)
        self.all_term()
        self.make_qubo_list()

    def cost_term(self, NUM_ITEM, value_list):
        # Cost term
        for a in range(NUM_ITEM):
            coef = -1 * value_list[a]
            self.qubo_cost[a, a] += coef

    def penalty_term(self, encoding, dim, NUM_ITEM, weight_list, max_weight_list, num_slack_list, ALPHA):
        if encoding == "one-hot":
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
                num_binary = math.floor(math.log(max_weight_list[dim_i]-1, 2))

                # (W_d)^2
                self.const_penalty[0] += ALPHA * max_weight_list[dim_i] * max_weight_list[dim_i]

                # -2 * W_d * ( \sum_(a=0)^(N-1) w_(d,a) x_a )
                for a in range(NUM_ITEM):
                    coef = -2 * max_weight_list[dim_i] * weight_list[dim_i, a]
                    idx = a
                    # print(f"-2 * {max_weight_list[dim_i]} * {weight_list[dim_i, a]} * x_{a} -> {coef}")
                    self.qubo_penalty[idx, idx] += ALPHA * coef

                # -2 * W_d * ( \sum_(n=0)^([log_2((W_d)-1)]) 2^n y_(d,n) )
                for n in range(num_binary+1):
                    coef = -2 * max_weight_list[dim_i] * pow(2, n)
                    idx = NUM_ITEM + sum(num_slack_list[:dim_i+1]) + n
                    # print(idx)
                    # print(f"-2 * {max_weight_list[dim_i]} * {pow(2, n)} * y_{dim_i, n} -> {coef}")
                    self.qubo_penalty[idx, idx] += ALPHA * coef

                #   ( \sum_(a=0)^(N-1) w_(d,a) x_a )^2
                # = \sum_(a=0)^(N-2) \sum_(b=a+1)^(N-1) 2 * w_(d,a) w_(d,b) x_a x_b + \sum_(a=0)^(N-1) (w_(d,a))^2 x_a
                # Quadratic term
                for a in range(NUM_ITEM-1):
                    for b in range(a+1, NUM_ITEM):
                        coef = 2 * weight_list[dim_i, a] * weight_list[dim_i, b]
                        idx_i = a
                        idx_j = b
                        # print(f"2 * {weight_list[dim_i, a]} * {weight_list[dim_i, b]} * x_{a} * x_{b} -> {coef}")
                        self.qubo_penalty[idx_i, idx_j] += ALPHA * coef
                # Linear term
                for a in range(NUM_ITEM):
                    coef = weight_list[dim_i, a] ** 2
                    idx = a
                    # print(f"{weight_list[dim_i, a]}^2 * x_{a} -> {coef}")
                    self.qubo_penalty[idx, idx] += ALPHA * coef

                #   2 * ( \sum_(a=0)^(N-1) w_(d,a) x_a) * ( \sum_(n=0)^([log_2((W_d)-1)]) 2^n y_(d,n) )
                # = \sum_(a=0)^(N-1) \sum_(n=0)^([log_2((W_d)-1)]) w_(d,a) 2^(n+1) x_a y_(d,n)
                # Quadratic term
                for a in range(NUM_ITEM):
                    for n in range(num_binary+1):
                        coef = weight_list[dim_i, a] * pow(2, n+1)
                        idx_i = a
                        idx_j = NUM_ITEM + sum(num_slack_list[:dim_i+1]) + n
                        # print(f"2 * {weight_list[dim_i, a]} * {pow(2, n+1)} * x_{a} * y_{dim_i, n} -> {coef}")
                        self.qubo_penalty[idx_i, idx_j] += ALPHA * coef

                #   ( \sum_(n=0)^([log_2((W_d)-1)]) 2^n y_(d,n) )^2
                # = \sum_(n=0)^([log_2((W_d)-1)]-1) \sum_(m=n+1)^([log_2((W_d)-1)]) 2 * 2^n 2^m y_(d,n) y_(d,m) + \sum_(n=0)^([log_2((W_d)-1)]) (2^n)^2 y_(d,n)
                # Quadratic term
                for n in range(num_binary):
                    for m in range(n+1, num_binary+1):
                        coef = 2 * pow(2, n) * pow(2, m)
                        idx_i = NUM_ITEM + sum(num_slack_list[:dim_i+1]) + n
                        idx_j = NUM_ITEM + sum(num_slack_list[:dim_i+1]) + m
                        # print(f"2 * {pow(2, n)} * {pow(2, m)} * y_{dim_i, n} * y_{dim_i, m} -> {coef}")
                        self.qubo_penalty[idx_i, idx_j] += ALPHA * coef
                # Linear term
                for n in range(num_binary+1):
                    coef = pow(2, n)**2
                    idx = NUM_ITEM + sum(num_slack_list[:dim_i+1]) + n
                    self.qubo_penalty[idx, idx] += ALPHA * coef
                    # print(f"{pow(2, n)**2} * y_{dim_i, n}-> {coef}")
