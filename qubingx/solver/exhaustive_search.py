from typing import Literal

import numpy.typing as npt

from qubingx.tools import QubingUtils

MODELTYPE = Literal["ISING", "QUBO"]

utils = QubingUtils()


def exhaustive_search(
    self,
    model_type: MODELTYPE,
    model: npt.NDArray,
    const: float = 0.0,
    hist: bool = False,
    output: bool = False,
) -> tuple:
    """Exhaustive search for Ising/QUBO model

    Args:
        model_type (Literal[&quot;ISING&quot;, &quot;QUBO&quot;]): Model type
        model (npt.NDArray): Ising/QUBO model
        const (float, optional): Constant value. Defaults to 0.0.
        hist (bool, optional): Return history. Defaults to False.
        output (bool, optional): Output the result. Defaults to False.

    Returns:
        tuple: Energy and solution
    """

    num_spin = len(model)
    energy_min = float("inf")
    solution_min = [0] * num_spin
    if hist:
        solution_list = []
        energy_list = []
    for i in range(2**num_spin):
        array = utils.decimal_to_binary_array(i, len(model))
        if model_type == "ISING":
            solution = utils.solution_qubo_to_ising(array)
            energy = utils.calc_energy(
                model_type="ISING", model=model, solution=solution, const=const
            )
        elif model_type == "QUBO":
            solution = array
            energy = utils.calc_energy(
                model_type="QUBO", model=model, solution=solution, const=const
            )
        else:
            raise ValueError("model_type must be 'ISING' or 'QUBO'")
        if output:
            print(f"spin: {solution}, energy: {energy}")
        if hist:
            solution_list.append(solution)
            energy_list.append(energy)
            if output:
                print(f"Update: spin: {solution_min}, energy: {energy_min}")
        if energy < energy_min:
            solution_min = solution
            energy_min = energy
    return (
        (energy_min, solution_min)
        if not hist
        else (energy_min, solution_min, energy_list, solution_list)
    )
