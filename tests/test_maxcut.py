import numpy as np
import pytest

from qubingx.cop.maxcut import MaxCut


@pytest.fixture
def adjacency_mtx():
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
def instance(adjacency_mtx):
    return MaxCut(adjacency_mtx=adjacency_mtx)


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
