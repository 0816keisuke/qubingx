from typing import Literal
import numpy as np
import dimod
import plotly.express as px

key_type = Literal["int", "str"]

def to_list(model_mtx):
    num_spin = len(model_mtx)
    model_list = []
    for idx_i in range(num_spin):
        for idx_j in range(idx_i, num_spin):
            coef = model_mtx[idx_i, idx_j]
            if not coef == 0:
                model_list.append([idx_i, idx_j, coef])
    return model_list


def to_dict(model_mtx, union: bool = True, key_type: key_type = "str", spin_name: str = ""):
    """Convert Ising/QUBO model to dict"""
    num_spin: int = len(model_mtx)
    if union:
        model_dict = {}
        # Linear & quadratic term
        for idx_i in range(num_spin):
            for idx_j in range(idx_i, num_spin):
                coef = model_mtx[idx_i, idx_j]
                if coef != 0:
                    if key_type == "int":
                        model_dict[(idx_i, idx_j)] = coef
                    elif key_type == "str":
                        model_dict[f"{spin_name}{idx_i}, {spin_name}{idx_j}"] = coef
        return model_dict

    else:
        model_dict_qd = {}
        model_dict_ln = {}
        # Linear term
        for idx_i in range(num_spin):
            if key_type == "int":
                model_dict_ln[idx_i] = model_mtx[idx_i, idx_i]
            elif key_type == "str":
                model_dict_ln[f"{spin_name}{idx_i}"] = model_mtx[idx_i, idx_i]
            # Quadratic term
            for idx_j in range(idx_i + 1, num_spin):
                coef = model_mtx[idx_i, idx_j]
                if coef != 0:
                    if key_type == "int":
                        model_dict_qd[(idx_i, idx_j)] = coef
                    elif key_type == "str":
                        model_dict_qd[f"{spin_name}{idx_i}, {spin_name}{idx_j}"] = coef
        return model_dict_ln, model_dict_qd


def to_bqm(model_mtx, MODEL: str, const: float = 0.0) -> dimod.BinaryQuadraticModel:
    """Convert Ising/QUBO model to BinaryQuadraticModel"""

    linear, quadratic = to_dict(model_mtx=model_mtx, union=False, key_type="int")

    if MODEL.upper() == "ISING":
        vartype = dimod.Vartype.SPIN
    elif MODEL.upper() == "QUBO":
        vartype = dimod.Vartype.BINARY
    else:
        msg = "Invalid Model Name"
        raise NameError(msg)

    return dimod.BinaryQuadraticModel(linear, quadratic, const, vartype)


# Calculate the hamiltonian's energy
def energy_qubo(model: np.ndarray, x: np.ndarray, const: float = 0.0) -> float:
    """Calculate an energy of QUBO model"""
    return float(np.dot(np.dot(x, model), x) + const)


def energy_ising(model: np.ndarray, x: np.ndarray, const: float = 0.0) -> float:
    """Calculate an energy of Ising model"""
    return float(np.dot(np.dot(x, np.triu(model, k=1)), x)) + np.dot(
        x, np.diag(model) + const
    )


def show(model: np.ndarray) -> None:
    """Show the matrix graph of the model"""

    fig = px.imshow(model)
    fig.show()


def ising_to_qubo(ising: np.ndarray, const: float) -> np.ndarray:
    """Convert QUBO model to Ising model"""
    num_spin = ising.shape[0]
    qubo = np.zeros((num_spin, num_spin), dtype=int)

    for i in range(num_spin):
        for j in range(i + 1, num_spin):
            qubo[i, j] = 4 * ising[i, j]
        qubo[i, i] = 2 * ising[i, i] - 4 * (
            ising[i, :].sum() + ising[:, i].sum() - 2 * qubo[i, i]
        )
    return qubo, ising.sum() - 2 * (np.diag(ising).sum()) + const


def qubo_to_ising(qubo: np.ndarray, const: float = 0) -> np.ndarray:
    """Convert QUBO model to Ising model"""
    num_spin = qubo.shape[0]
    ising = np.zeros((num_spin, num_spin), dtype=float)

    for i in range(num_spin):
        for j in range(i + 1, num_spin):
            ising[i, j] = qubo[i, j] / 4
        ising[i, i] = (qubo[i, :].sum() + qubo[:, i].sum() - qubo[i, i]) / 2
    return (
        ising,
        (qubo.sum() - np.diag(qubo).sum()) / 4 + np.diag(qubo).sum() / 2 + const,
    )
