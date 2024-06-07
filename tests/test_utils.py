import numpy as np
import pytest

from qubingx.tools import QubingUtils


@pytest.fixture
def utils():
    return QubingUtils()


@pytest.fixture
# QubingUtils().random_model(n=5, v_min=-9, v_max=9, seed=0)で生成したQUBO行列
def ising_mtx():
    return np.array(
        [
            [3, 6, -9, -6, -6],
            [0, 0, 9, -5, -3],
            [0, 0, -3, -2, 5],
            [0, 0, 0, -1, 0],
            [0, 0, 0, 0, -9],
        ]
    )


@pytest.fixture
# List ising_mtx()
def ising_list():
    return [
        [0, 0, 3],
        [0, 1, 6],
        [0, 2, -9],
        [0, 3, -6],
        [0, 4, -6],
        [1, 2, 9],
        [1, 3, -5],
        [1, 4, -3],
        [2, 2, -3],
        [2, 3, -2],
        [2, 4, 5],
        [3, 3, -1],
        [4, 4, -9],
    ]


@pytest.fixture
# Dict ising_mtx()
def ising_dict():
    return {
        (0, 0): 3,
        (0, 1): 6,
        (0, 2): -9,
        (0, 3): -6,
        (0, 4): -6,
        (1, 2): 9,
        (1, 3): -5,
        (1, 4): -3,
        (2, 2): -3,
        (2, 3): -2,
        (2, 4): 5,
        (3, 3): -1,
        (4, 4): -9,
    }


@pytest.fixture
def ising_dict_ln():
    return {(0, 0): 3, (2, 2): -3, (3, 3): -1, (4, 4): -9}


@pytest.fixture
def ising_dict_qd():
    return {
        (0, 1): 6,
        (0, 2): -9,
        (0, 3): -6,
        (0, 4): -6,
        (1, 2): 9,
        (1, 3): -5,
        (1, 4): -3,
        (2, 3): -2,
        (2, 4): 5,
    }


@pytest.fixture
# QubingUtils().random_model(n=5, v_min=-9, v_max=9, seed=1)で生成したQUBO行列
def qubo_mtx():
    return np.array(
        [[-4, 2, 3, -1, 0], [0, -4, 6, -9, 7], [0, 0, -2, 4, -3], [0, 0, 0, 2, 1], [0, 0, 0, 0, 8]]
    )


@pytest.fixture
# List qubo_mtx()
def qubo_list():
    return [
        [0, 0, -4],
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, -1],
        [1, 1, -4],
        [1, 2, 6],
        [1, 3, -9],
        [1, 4, 7],
        [2, 2, -2],
        [2, 3, 4],
        [2, 4, -3],
        [3, 3, 2],
        [3, 4, 1],
        [4, 4, 8],
    ]


@pytest.fixture
# Dict qubo_mtx()
def qubo_dict():
    return {
        (0, 0): -4,
        (0, 1): 2,
        (0, 2): 3,
        (0, 3): -1,
        (1, 1): -4,
        (1, 2): 6,
        (1, 3): -9,
        (1, 4): 7,
        (2, 2): -2,
        (2, 3): 4,
        (2, 4): -3,
        (3, 3): 2,
        (3, 4): 1,
        (4, 4): 8,
    }


@pytest.fixture
def qubo_dict_ln():
    return {(0, 0): -4, (1, 1): -4, (2, 2): -2, (3, 3): 2, (4, 4): 8}


@pytest.fixture
def qubo_dict_qd():
    return {
        (0, 1): 2,
        (0, 2): 3,
        (0, 3): -1,
        (1, 2): 6,
        (1, 3): -9,
        (1, 4): 7,
        (2, 3): 4,
        (2, 4): -3,
        (3, 4): 1,
    }


@pytest.fixture
def solution_ising_opt():
    return np.array([-1, 1, -1, -1, 1])  # ising_mtx()の最適解


@pytest.fixture
def solution_qubo_opt():
    return np.array([1, 1, 0, 1, 0])  # qubo_mtx()の最適解


@pytest.fixture
# QubingUtils().random_solution(n=5, model_type="ISING", seed=0)で生成した解
def solution_ising():
    return np.array([1, 1, -1, -1, 1])


@pytest.fixture
# QubingUtils().random_solution(n=5, model_type="QUBO", seed=0)で生成した解
def solution_qubo():
    return np.array([1, 1, 0, 0, 1])


@pytest.fixture
# QubingUtils().random_assignment(n=5, seed=0)で生成した解
def assignment():
    return np.array([3, 4, 0, 1, 2])


@pytest.fixture
# QubingUtils().random_twoway_onehot_solution(n=5, mtx=True, seed=0)で生成した解
def solution_twoway_onehot():
    return np.array(
        [[0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]
    )


def test_calc_energy(utils, ising_mtx, qubo_mtx, solution_ising_opt, solution_qubo_opt):
    # ISINGモデル
    expected_energy = -37.0
    energy = utils.calc_energy(model_type="ISING", model=ising_mtx, solution=solution_ising_opt)
    assert energy == expected_energy

    # QUBOモデル
    expected_energy = -14.0
    energy = utils.calc_energy(model_type="QUBO", model=qubo_mtx, solution=solution_qubo_opt)
    assert energy == expected_energy


def test_decimal_to_binary_array(utils):
    assert utils.decimal_to_binary_array(num_decimal=10, n=4) == [1, 0, 1, 0]


def test_random_model(utils, ising_mtx, qubo_mtx):
    # ISINGモデル
    assert (utils.random_model(n=5, v_min=-9, v_max=9, seed=0) == ising_mtx).all()

    # QUBOモデル
    assert (utils.random_model(n=5, v_min=-9, v_max=9, seed=1) == qubo_mtx).all()


def test_random_solution(utils, solution_ising, solution_qubo):
    # ISINGモデル
    assert (utils.random_solution(n=5, model_type="ISING", seed=0) == solution_ising).all()

    # QUBOモデル
    assert (utils.random_solution(n=5, model_type="QUBO", seed=0) == solution_qubo).all()


def test_random_twoway_onehot_solution(utils, solution_twoway_onehot):
    assert (
        utils.random_twoway_onehot_solution(n=5, mtx=True, seed=0) == solution_twoway_onehot
    ).all()
    assert (
        utils.random_twoway_onehot_solution(n=5, mtx=False, seed=0)
        == np.ravel(solution_twoway_onehot)
    ).all()


def test_random_assignment(utils, assignment):
    assert (utils.random_assignment(n=5, seed=0) == assignment).all()


def test_solution_qubo_to_ising(utils, solution_qubo, solution_ising):
    assert (utils.solution_qubo_to_ising(solution_qubo) == solution_ising).all()


def test_solution_ising_to_qubo(utils, solution_ising, solution_qubo):
    assert (utils.solution_ising_to_qubo(solution_ising) == solution_qubo).all()


def test_qubo_solution_to_assignment(utils, solution_twoway_onehot):
    expected = [2, 3, 4, 0, 1]
    actual = utils.qubo_solution_to_assignment(np.ravel(solution_twoway_onehot))
    assert actual == expected


def test_exhaustive_search(utils, ising_mtx, qubo_mtx):
    # ISINGモデル
    expected_energy = -37.0
    expected_solution = np.array([-1, 1, -1, -1, 1])
    energy, solution = utils.exhaustive_search(model=ising_mtx, model_type="ISING")
    assert energy == expected_energy
    assert (solution == expected_solution).all()

    # QUBOモデル
    expected_energy = -14.0
    expected_solution = np.array([1, 1, 0, 1, 0])
    energy, solution = utils.exhaustive_search(model=qubo_mtx, model_type="QUBO")
    assert energy == expected_energy
    assert (solution == expected_solution).all()


def test_mtx_to_list(utils, ising_mtx, ising_list, qubo_mtx, qubo_list):
    # ISINGモデル
    assert ising_list == utils.mtx_to_list(model=ising_mtx)
    # QUBOモデル
    assert qubo_list == utils.mtx_to_list(model=qubo_mtx)


def test_mtx_to_dict(utils, ising_mtx, ising_dict, qubo_mtx, qubo_dict):
    # ISINGモデル
    assert ising_dict == utils.mtx_to_dict(model=ising_mtx)
    # QUBOモデル
    assert qubo_dict == utils.mtx_to_dict(model=qubo_mtx)


def test_to_dict_all(utils, ising_mtx, ising_dict, qubo_mtx, qubo_dict):
    # ISINGモデル
    assert ising_dict == utils._to_dict_all(model=ising_mtx, key_type="int")
    # QUBOモデル
    assert qubo_dict == utils._to_dict_all(model=qubo_mtx, key_type="int")


def test_to_dict_qd_ln(
    utils, ising_mtx, ising_dict_qd, ising_dict_ln, qubo_mtx, qubo_dict_qd, qubo_dict_ln
):
    # ISINGモデル
    assert ising_dict_ln, ising_dict_qd == utils._to_dict_qd_ln(model=ising_mtx, key_type="int")
    # QUBOモデル
    assert qubo_dict_ln, qubo_dict_qd == utils._to_dict_qd_ln(model=qubo_mtx, key_type="int")
