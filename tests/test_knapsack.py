# import numpy as np

# from qubingx.knapsack import Knapsack

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


import numpy as np
import pytest

from qubingx.cop.knapsack import Knapsack


@pytest.fixture
def values():
    return np.array([5, 7, 2, 1, 4, 3])


@pytest.fixture
def weights():
    return np.array([8, 10, 6, 4, 5, 3])


@pytest.fixture
def max_weight():
    return 20


@pytest.fixture
def instance(values, weights, max_weight):
    return Knapsack(values=values, weights=weights, max_weight=max_weight, alpha=10)


def test_h_obj_onehot(instance):
    expected_q_obj = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    assert np.array_equal(instance.q_obj, expected_q_obj)


def test_h_constraint_onehot(instance):
    expected_q_constraint = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    assert np.array_equal(instance.q_constraint, expected_q_constraint)


def test_h_obj_binary(instance):
    expected_q_obj = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    assert np.array_equal(instance.q_obj, expected_q_obj)


def test_h_constraint_binary(instance):
    expected_q_constraint = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    assert np.array_equal(instance.q_constraint, expected_q_constraint)
