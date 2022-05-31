import numpy as np
from qubox.knapsack import Knapsack

# https://tutorial.openjij.org/build/html/ja/008-KnapsackPyqubo.html
# value_list = np.array([5, 7, 2, 1, 4, 3])
# weight_list = np.array([8, 10, 6, 4, 5, 3])
# max_weight = 20
# solution_onehot_enc = np.array([0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
# solution_log_enc = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0])

# The bellow is the original data
# value_list = np.array([2, 2, 3, 4])
# weight_list = np.array([1, 2, 3, 4])
# max_weight = 7
# solution_onehot_enc = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0])
# solution_onehot_enc = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1])
# solution_log_enc = np.array([1, 1, 1, 0, 1, 0, 0])
# solution_log_enc = np.array([0, 0, 1, 1, 0, 0, 0])