import numpy as np
from abc import ABCMeta, abstractmethod

class Base(metaclass=ABCMeta):
    def __init__(self, num_spin):
        # QUBO matrix
        self.num_spin = num_spin
        self.q_all = np.zeros((self.num_spin, self.num_spin))
        self.q_cost = np.zeros((self.num_spin, self.num_spin))
        self.q_pen = np.zeros((self.num_spin, self.num_spin))

        self.const_all = np.zeros((1))
        self.const_cost = np.zeros((1))
        self.const_pen = np.zeros((1))

    @abstractmethod
    def h_cost(self):
        pass

    @abstractmethod
    def h_pen(self):
        pass

    def h_all(self):
        self.q_all = self.q_cost + self.q_pen
        self.const_all = self.const_cost + self.const_pen

    # Convert QUBO-model and coefficient to list
    def to_list(self, Q, const=0):
        Q_list = []
        for i in range(len(Q)):
            for j in range(i, len(Q)):
                coef = Q[i, j]
                if not coef == 0:
                    Q_list.append([i, j, coef])
        if const != 0:
            Q_list.append([-1, -1, int(const)])
        return Q_list

    # def convert_qubo_to_ising(self):
    #     self.const_ising += 4 * self.const_qubo # Multiply by 4 to convert to integer, as appearance of fraction 1/4
    #     for i in range(self.num_spin):
    #         self.const_ising += 2 * self.bias_qubo[i]
    #         self.bias_ising[i] += 2 * self.bias_qubo[i]
    #         for j in range(i, self.num_spin):
    #             self.const_ising += self.weight_qubo[i][j]
    #             self.bias_ising[i] += self.weight_qubo[i][j]
    #             self.bias_ising[j] += self.weight_qubo[i][j]
    #             self.weight_ising[i][j] += self.weight_qubo[i][j]
    #             # 下三角を埋める
    #             self.weight_ising[j][i] = self.weight_ising[i][j]

    # def convert_ising_to_qubo(self):
    #     self.const_qubo += self.const_ising
    #     for i in range(self.num_spin):
    #         self.const_qubo += -1 * self.bias_ising[i]
    #         self.bias_qubo[i] += 2 * self.bias_ising[i]
    #         for j in range(i, self.num_spin):
    #             self.const_qubo += self.weight_ising[i][j]
    #             self.bias_qubo[i] += -2 * self.weight_ising[i][j]
    #             self.bias_qubo[j] += -2 * self.weight_ising[i][j]
    #             self.weight_qubo[i][j] += 4 * self.weight_ising[i][j]
    #             # 下三角を埋める
    #             self.weight_qubo[j][i] = self.weight_qubo[i][j]
