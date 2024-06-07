from typing import Literal

import numpy.typing as npt

from qubingx.tools import QubingUtils

MODELTYPE = Literal["ISING", "QUBO"]

utils = QubingUtils()


def random_search(
    self,
    model_type: MODELTYPE,
    model: npt.NDArray,
    const: float = 0.0,
    num_iter: int = 1000,
    init_solution: list[int] | npt.NDArray | None = None,
    hist: bool = False,
    output: bool = False,
) -> tuple:
    """Random search for Ising/QUBO model

    Args:
        model_type (Literal[&quot;ISING&quot;, &quot;QUBO&quot;]): Model type
        model (npt.NDArray): Ising/QUBO model
        const (float, optional): Constant value. Defaults to 0.0.
        num_iter (int, optional): Number of iterations. Defaults to 1000.
        init_solution (list[int] | npt.NDArray | None, optional): Initial solution. Defaults to None.
        hist (bool, optional): Return history. Defaults to False.
        output (bool, optional): Output the result. Defaults to False.

    Returns:
        tuple: Energy and solution
    """
    num_spin = len(model)
    if init_solution is None:
        init_solution = utils.random_solution(n=num_spin, model_type=model_type)
    energy_min = utils.calc_energy(
        model_type=model_type, model=model, solution=init_solution, const=const
    )
    solution_min = init_solution
    if hist:
        solution_list = [solution_min]
        energy_list = [energy_min]
    for _ in range(num_iter):
        solution = utils.random_solution(n=num_spin, model_type=model_type)
        energy = utils.calc_energy(
            model_type=model_type, model=model, solution=solution, const=const
        )
        if energy < energy_min:
            solution_min = solution
            energy_min = energy
            if hist:
                solution_list.append(solution_min)
                energy_list.append(energy_min)
            if output:
                print(f"spin: {solution_min}, energy: {energy_min}")
    return (
        (energy_min, solution_min)
        if not hist
        else (energy_min, solution_min, energy_list, solution_list)
    )
