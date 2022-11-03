import numpy as np

from qubox.maxcut import MaxCut

# # This test is cited from arXiv, the followiong link:
# # https://arxiv.org/abs/1811.11538
adjacency_matrix = np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 0], [1, 0, 0, 1, 1], [0, 1, 1, 0, 1], [0, 0, 1, 1, 0]])
instance = MaxCut(adjacency_matrix=adjacency_matrix)


def test_qubo_matrix_cost_term():
    expect = np.array(
        [
            [-2.0, 2.0, 2.0, 0.0, 0.0],
            [0.0, -2.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, -3.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, -3.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, -2.0],
        ]
    )
    result = instance.qubo_cost
    np.testing.assert_array_equal(expect, result)
    # assert (expect == result).all()


def test_qubo_matrix_penalty_term():
    expect = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    result = instance.qubo_penalty
    np.testing.assert_array_equal(expect, result)


def test_qubo_matrix_all_term():
    expect = np.array(
        [
            [-2.0, 2.0, 2.0, 0.0, 0.0],
            [0.0, -2.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, -3.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, -3.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, -2.0],
        ]
    )
    result = instance.qubo_all
    np.testing.assert_array_equal(expect, result)


def test_const_cost_term():
    expect = np.array(0)
    result = instance.const_cost
    np.testing.assert_array_equal(expect, result)


def test_const_penalty_term():
    expect = np.array(0)
    result = instance.const_penalty
    np.testing.assert_array_equal(expect, result)


def test_const_all_term():
    expect = np.array(0)
    result = instance.const_all
    np.testing.assert_array_equal(expect, result)


def test_qubo_list_cost_term():
    expect = np.array(
        [
            [0, 0, -2.0],
            [0, 1, 2.0],
            [0, 2, 2.0],
            [1, 1, -2.0],
            [1, 3, 2.0],
            [2, 2, -3.0],
            [2, 3, 2.0],
            [2, 4, 2.0],
            [3, 3, -3.0],
            [3, 4, 2.0],
            [4, 4, -2.0],
        ]
    )
    result = instance.qubo_list_cost
    np.testing.assert_array_equal(expect, result)


def test_qubo_list_penalty_term():
    expect = np.array([])
    result = instance.qubo_list_penalty
    np.testing.assert_array_equal(expect, result)


def test_qubo_list_all_term():
    expect = np.array(
        [
            [0, 0, -2.0],
            [0, 1, 2.0],
            [0, 2, 2.0],
            [1, 1, -2.0],
            [1, 3, 2.0],
            [2, 2, -3.0],
            [2, 3, 2.0],
            [2, 4, 2.0],
            [3, 3, -3.0],
            [3, 4, 2.0],
            [4, 4, -2.0],
        ]
    )
    result = instance.qubo_list_all
    np.testing.assert_array_equal(expect, result)
