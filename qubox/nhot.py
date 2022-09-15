import numpy as np
from qubox.base import Base

class NHot(Base):
    def __init__(self,
                num_spin_row,
                hot_num=1,
                row_hot=True,
                col_hot=True,
                ALPHA=1
                ):
        # Check tye type of Arguments
        if not isinstance(num_spin_row, int):
            print("The type of the argument 'num_spin_row' is WRONG.")
            print("It shoud be int.")
            exit()
        if not isinstance(hot_num, int):
            print("The type of the argument 'hot_num' is WRONG.")
            print("It shoud be int.")
            exit()
        if not isinstance(row_hot, bool):
            print("The type of the argument 'row_hot' is WRONG.")
            print("It shoud be bool.")
            exit()
        if not isinstance(col_hot, bool):
            print("The type of the argument 'col_hot' is WRONG.")
            print("It shoud be bool.")
            exit()

        super().__init__(num_spin = num_spin_row * num_spin_row)
        self.spin_index = np.arange(num_spin_row * num_spin_row).reshape(num_spin_row, num_spin_row)
        np.set_printoptions(edgeitems=10) # Chenge the setting for printing numpy

        self.cost_term()
        self.penalty_term(num_spin_row, hot_num, row_hot, col_hot, ALPHA)
        self.all_term()
        self.make_qubo_list()

    def cost_term(self):
        pass

    def penalty_term(
        self,
        num_spin_row,
        hot_num,
        row_hot,
        col_hot,
        ALPHA
        ):
        if row_hot:
            # Constraint term1 (1-hot of horizontal line)
            # Quadratic term
            for i in range(num_spin_row):
                for k in range(num_spin_row-1):
                    for l in range(k+1, num_spin_row):
                        idx_i = self.spin_index[i, k]
                        idx_j = self.spin_index[i, l]
                        coef = 2
                        self.qubo_penalty[idx_i, idx_j] += ALPHA * coef
            # Linear term
            for i in range(num_spin_row):
                for k in range(num_spin_row):
                    idx = self.spin_index[i, k]
                    coef = 1 - 2 * hot_num
                    self.qubo_penalty[idx, idx] += ALPHA * coef
            # Constant term
            self.const_penalty[0] += ALPHA * hot_num**2 * num_spin_row

        if col_hot:
            # Constraint term2 (1-hot of vertical line)
            # Quadratic term
            for k in range(num_spin_row):
                for i in range(num_spin_row-1):
                    for j in range(i+1, num_spin_row):
                        idx_i = self.spin_index[i, k]
                        idx_j = self.spin_index[j, k]
                        coef = 2
                        self.qubo_penalty[idx_i, idx_j] += ALPHA * coef
            # Linear term
            for k in range(num_spin_row):
                for i in range(num_spin_row):
                    idx = self.spin_index[i, k]
                    coef = 1 - 2 * hot_num
                    self.qubo_penalty[idx, idx] += ALPHA * coef
            # Constant term
            self.const_penalty[0] += ALPHA * hot_num**2 * num_spin_row
