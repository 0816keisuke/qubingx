import numpy as np
import pytest

from qubingx.cop.tsp import TSP


@pytest.fixture
def distance_mtx():
    return np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )


@pytest.fixture
def instance(distance_mtx):
    return TSP(distance_mtx=distance_mtx, alpha=10)


def test_h_obj(instance):
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


def test_h_constraint(instance):
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
