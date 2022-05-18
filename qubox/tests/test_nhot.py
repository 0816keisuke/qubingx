import numpy as np
from qubox.nhot import NHot

num_spin_row = 4

def make_qubo_model(num_spin_row, hot_num, row_hot, col_hot):
    num_spin = num_spin_row * num_spin_row
    qubo_penalty = np.zeros((num_spin, num_spin))
    # Row
    if row_hot:
        for start_point in range(0, num_spin, num_spin_row):
            end_point = start_point + num_spin_row
            for i in range(start_point, end_point):
                qubo_penalty[i, i] += 1 - 2 * hot_num
                for j in range(i+1, end_point):
                    qubo_penalty[i, j] += 2

    # Col
    if col_hot:
        for start_point in range(num_spin_row):
            for i in range(start_point, num_spin, num_spin_row):
                qubo_penalty[i, i] += 1 - 2 * hot_num
                for j in range(i+num_spin_row, num_spin, num_spin_row):
                    qubo_penalty[i, j] += 2

    return qubo_penalty


def test_row_one_hot():
    instance = NHot(num_spin_row=num_spin_row, hot_num=1, row_hot=True, col_hot=False)
    expect = make_qubo_model(num_spin_row, hot_num=1, row_hot=True, col_hot=False)
    result = instance.qubo_penalty
    np.testing.assert_array_equal(expect, result)

def test_col_one_hot():
    instance = NHot(num_spin_row=num_spin_row, hot_num=1, row_hot=False, col_hot=True)
    expect = make_qubo_model(num_spin_row, hot_num=1, row_hot=False, col_hot=True)
    result = instance.qubo_penalty
    np.testing.assert_array_equal(expect, result)

def test_row_col_one_hot():
    instance = NHot(num_spin_row=num_spin_row, hot_num=1, row_hot=True, col_hot=True)
    expect = make_qubo_model(num_spin_row, hot_num=1, row_hot=True, col_hot=True)
    result = instance.qubo_penalty
    np.testing.assert_array_equal(expect, result)

def test_row_two_hot():
    instance = NHot(num_spin_row=num_spin_row, hot_num=2, row_hot=True, col_hot=False)
    expect = make_qubo_model(num_spin_row, hot_num=2, row_hot=True, col_hot=False)
    result = instance.qubo_penalty
    np.testing.assert_array_equal(expect, result)

def test_col_two_hot():
    instance = NHot(num_spin_row=num_spin_row, hot_num=2, row_hot=False, col_hot=True)
    expect = make_qubo_model(num_spin_row, hot_num=2, row_hot=False, col_hot=True)
    result = instance.qubo_penalty
    np.testing.assert_array_equal(expect, result)

def test_row_col_two_hot():
    instance = NHot(num_spin_row=num_spin_row, hot_num=2, row_hot=True, col_hot=True)
    expect = make_qubo_model(num_spin_row, hot_num=2, row_hot=True, col_hot=True)
    result = instance.qubo_penalty
    np.testing.assert_array_equal(expect, result)
