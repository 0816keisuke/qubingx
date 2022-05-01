import numpy as np
from abc import ABCMeta, abstractmethod
# import plotly.express as px

class Base(metaclass=ABCMeta):
    def __init__(self, NUM_SPIN):
        # QUBO matrix
        self.NUM_SPIN = NUM_SPIN
        self.qubo_all = np.zeros((self.NUM_SPIN, self.NUM_SPIN))
        self.qubo_cost = np.zeros((self.NUM_SPIN, self.NUM_SPIN))
        self.qubo_penalty = np.zeros((self.NUM_SPIN, self.NUM_SPIN))

        self.const_all = np.zeros((1))
        self.const_cost = np.zeros((1))
        self.const_penalty = np.zeros((1))

        # QUBO index and coefficient list
        self.qubo_list_all = []
        self.qubo_list_cost = []
        self.qubo_list_penalty = []

    @abstractmethod
    def cost_term(self):
        pass

    @abstractmethod
    def penalty_term(self):
        pass

    def all_term(self):
        self.qubo_all = self.qubo_cost + self.qubo_penalty
        self.const_all = self.const_cost + self.const_penalty

    # Make QUBO index and coefficient list
    def make_qubo_list(self):
        # Cost term
        for i in range(self.NUM_SPIN):
            for j in range(i, self.NUM_SPIN):
                coef = self.qubo_cost[i, j]
                if not coef == 0:
                    self.qubo_list_cost.append([i, j, coef])
        if not self.const_cost[0] == 0:
            self.qubo_list_cost.append([-1, -1, self.const_cost[0]])
        self.qubo_list_cost = np.array(self.qubo_list_cost)

        # Penalty term
        for i in range(self.NUM_SPIN):
            for j in range(i, self.NUM_SPIN):
                coef = self.qubo_penalty[i, j]
                if not coef == 0:
                    self.qubo_list_penalty.append([i, j, coef])
        if not self.const_penalty[0] == 0:
            self.qubo_list_penalty.append([-1, -1, self.const_penalty[0]])
        self.qubo_list_penalty = np.array(self.qubo_list_penalty)

        # All term
        for i in range(self.NUM_SPIN):
            for j in range(i, self.NUM_SPIN):
                coef = self.qubo_all[i, j]
                if not coef == 0:
                    self.qubo_list_all.append([i, j, coef])
        if not self.const_all[0] == 0:
            self.qubo_list_all.append([-1, -1, self.const_all[0]])
        self.qubo_list_all = np.array(self.qubo_list_all)

    # def convert_qubo_to_ising(self):
    #     self.const_ising += 4 * self.const_qubo # Multiply by 4 to convert to integer, as appearance of fraction 1/4
    #     for i in range(self.NUM_SPIN):
    #         self.const_ising += 2 * self.bias_qubo[i]
    #         self.bias_ising[i] += 2 * self.bias_qubo[i]
    #         for j in range(i, self.NUM_SPIN):
    #             self.const_ising += self.weight_qubo[i][j]
    #             self.bias_ising[i] += self.weight_qubo[i][j]
    #             self.bias_ising[j] += self.weight_qubo[i][j]
    #             self.weight_ising[i][j] += self.weight_qubo[i][j]
    #             # 下三角を埋める
    #             self.weight_ising[j][i] = self.weight_ising[i][j]

    # def convert_ising_to_qubo(self):
    #     self.const_qubo += self.const_ising
    #     for i in range(self.NUM_SPIN):
    #         self.const_qubo += -1 * self.bias_ising[i]
    #         self.bias_qubo[i] += 2 * self.bias_ising[i]
    #         for j in range(i, self.NUM_SPIN):
    #             self.const_qubo += self.weight_ising[i][j]
    #             self.bias_qubo[i] += -2 * self.weight_ising[i][j]
    #             self.bias_qubo[j] += -2 * self.weight_ising[i][j]
    #             self.weight_qubo[i][j] += 4 * self.weight_ising[i][j]
    #             # 下三角を埋める
    #             self.weight_qubo[j][i] = self.weight_qubo[i][j]
