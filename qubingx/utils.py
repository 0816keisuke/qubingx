import random
from typing import Literal

import dimod
import numpy as np
import numpy.typing as npt
import plotly.express as px

MODELTYPE = Literal["ISING", "QUBO"]
KEYTYPE = Literal["int", "str"]


class QubingUtils:
    def __init__(self) -> None:
        pass

    def calc_energy(
        self, model_type: MODELTYPE, model: npt.NDArray, solution: npt.NDArray, const: float = 0.0
    ) -> float:
        """Calculate an energy of Ising/QUBO model

        Args:
            model (npt.NDArray): Ising/QUBO model
            x (npt.NDArray): Solution
            const (float, optional): Constant value. Defaults to 0.0.

        Returns:
            float: Energy value of the Ising/QUBO model
        """
        if model_type == "ISING":
            return float(np.dot(np.dot(solution, np.triu(model, k=1)), solution)) + np.dot(
                solution, np.diag(model) + const
            )
        elif model_type == "QUBO":
            return float(np.dot(solution, np.dot(model, solution)) + const)
        else:
            raise ValueError("model_type must be 'ISING' or 'QUBO'")

    def check_constraint(
        self, model_type: MODELTYPE, model: npt.NDArray, solution: npt.NDArray, const: float = 0.0
    ) -> bool:
        """Check if the solution satisfies the constraint

        Args:
            model_type (Literal[&quot;ISING&quot;, &quot;QUBO&quot;]): Model type
            model (npt.NDArray): Ising/QUBO model
            solution (npt.NDArray): Solution

        Returns:
            bool: Return True if the solution satisfies the constraint.
                  Return False if the solution does not satisfy the constraint.
        """
        return (
            True
            if self.calc_energy(model_type=model_type, model=model, solution=solution, const=const)
            == 0.0
            else False
        )

    def decimal_to_binary_array(self, num_decimal: int, n: int) -> list[int]:
        """Convert decimal number to binary array

        Args:
            num_decimal (int): Decimal number
            n (int): Length of the binary array

        Returns:
            npt.NDArray: Binary array
        """
        binary_str = bin(num_decimal)[2:].zfill(n)
        return [int(bit) for bit in binary_str]

    def random_model(
        self, n: int, v_min: int = -5, v_max: int = 5, seed: int | None = None
    ) -> npt.NDArray:
        """Generate random Ising/QUBO model with n spins and value range [v_min, v_max]

        Args:
            n (int): Number of spins
            v_min (int, optional): Minimum value of the model. Defaults to -5.
            v_max (int, optional): Maximum value of the model. Defaults to 5.
            seed (int, optional): Random seed. Defaults to None.

        Returns:
            npt.NDArray: Random Ising/QUBO model
        """
        if seed is not None:
            np.random.seed(seed)
        model = np.random.randint(v_min, v_max + 1, (n, n))
        model = np.triu(model, k=0)
        return model

    def random_solution(self, n: int, model_type: MODELTYPE, seed: int | None = None) -> list[int]:
        """Generate random Ising/QUBO model solution

        Args:
            n (int): Number of spins
            model_type (Literal[&quot;ISING&quot;, &quot;QUBO&quot;]): Model type
            seed (int | None, optional): Random seed. Defaults to None.

        Returns:
            list[int]: Random Ising/QUBO model solution
        """
        if model_type not in ["ISING", "QUBO"]:
            raise ValueError("model_type must be 'ISING' or 'QUBO'")
        if seed is not None:
            random.seed(seed)
        return (
            random.choices([-1, 1], k=n) if model_type == "ISING" else random.choices([0, 1], k=n)
        )

    # 2way-1hot制約を有するn行n列のランダムなQUBOの解を生成する
    def random_twoway_onehot_solution(
        self, n: int, mtx: bool = False, seed: int | None = None
    ) -> list[int]:
        """Generate random QUBO model solution with 2-way-1hot constraint

        Args:
            n (int): Number of rows and columns
            seed (int | None, optional): Random seed. Defaults to None.
            mtx (bool, optional): Matrix format. Defaults to False.

        Returns:
            npt.NDArray: Random QUBO model solution with 2-way-1hot constraint
        """
        if seed is not None:
            random.seed(seed)
        assignment = self.random_assignment(n=n, seed=seed)
        return self.assignment_to_qubo_solution(assignment=assignment, mtx=mtx)

    def random_assignment(self, n: int, seed: int | None = None) -> list[int]:
        """Generate random assignment like TSP or QAP

        Args:
            n (int): Number of spins
            seed (int | None, optional): Random seed. Defaults to None.

        Returns:
            list[int]: Random assignment
        """
        if seed is not None:
            random.seed(seed)
        return random.sample(range(n), n)

    def solution_ising_to_qubo(self, solution: npt.NDArray) -> npt.NDArray:
        """Convert Ising model solution to QUBO model solution

        Args:
            solution (npt.NDArray): Ising model solution

        Returns:
            npt.NDArray: QUBO model solution
        """
        return [(x + 1) // 2 for x in solution]

    def solution_qubo_to_ising(self, solution: npt.NDArray) -> npt.NDArray:
        """Convert QUBO model solution to Ising model solution

        Args:
            solution (npt.NDArray): QUBO model solution

        Returns:
            npt.NDArray: Ising model solution
        """
        return [2 * x - 1 for x in solution]

    def assignment_to_qubo_solution(
        self, assignment: list[int] | npt.NDArray, mtx: bool = False
    ) -> list[int]:
        """Generate QUBO model solution from assignment

        Args:
            assignment (list[int] | npt.NDArray): Assignment
            mtx (bool, optional): Matrix format. Defaults to False.

        Returns:
            npt.NDArray: QUBO model solution
        """
        num_spin = len(assignment)
        x_m = np.zeros((num_spin, num_spin), dtype=np.int8)
        for i, j in enumerate(assignment):
            x_m[i][j - 1] = 1
        return x_m.tolist() if mtx else np.ravel(x_m).tolist()

    def qubo_solution_to_assignment(self, solution: npt.NDArray) -> list[int]:
        """Generate assignment from QUBO model solution

        Args:
            solution (npt.NDArray): QUBO model solution

        Returns:
            list[int]: Assignment
        """
        num_spin = int(len(solution) ** 0.5)
        x_m = np.reshape(solution, (num_spin, num_spin))
        return [np.where(x_m[i] == 1)[0][0] for i in range(num_spin)]

    def exhaustive_search(
        self,
        model_type: MODELTYPE,
        model: npt.NDArray,
        const: float = 0.0,
        hist: bool = False,
        output: bool = False,
    ) -> tuple:
        num_spin = len(model)
        energy_min = float("inf")
        solution_min = [0] * num_spin
        if hist:
            solution_list = []
            energy_list = []
        for i in range(2**num_spin):
            array = self.decimal_to_binary_array(i, len(model))
            if model_type == "ISING":
                solution = self.solution_qubo_to_ising(array)
                energy = self.calc_energy(
                    model_type="ISING", model=model, solution=solution, const=const
                )
            elif model_type == "QUBO":
                solution = array
                energy = self.calc_energy(
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

    # Isingモデル/QUBOモデルをリスト形式に変換
    def mtx_to_list(self, model: npt.NDArray) -> list[list[int | float]]:
        """Convert Ising/QUBO model matrix to list

        Args:
            model (npt.NDArray): Ising/QUBO model matrix

        Returns:
            list[list[int | float]]: Ising/QUBO model list
        """
        num_spin = len(model)
        model_list = []
        for idx_i in range(num_spin):
            for idx_j in range(idx_i, num_spin):
                coef = model[idx_i, idx_j]
                if not coef == 0:
                    model_list.append([idx_i, idx_j, coef])
        return model_list

    # Isingモデル/QUBOモデルを辞書形式に変換
    # TODO: 返り値の型ヒントを追加
    def mtx_to_dict(
        self,
        model: npt.NDArray,
        union: bool = True,
        key_type: KEYTYPE = "int",
        spin_name: str | None = None,
    ):
        """Convert Ising/QUBO model to dictionary

        Args:
            model (npt.NDArray): Ising/QUBO model
            union (bool, optional): Union of linear and quadratic terms. Defaults to True.
            key_type (Literal[&quot;int&quot;, &quot;str&quot;], optional): Key type of the dictionary. Defaults to &quot;int&quot;.
            spin_name (str | None, optional): Spin name. Defaults to None.

        Returns:
            _type_: _description_
        """
        if union:
            return self._to_dict_all(model, key_type, spin_name)
        else:
            return self._to_dict_qd_ln(model, key_type, spin_name)

    def _to_dict_all(
        self,
        model: npt.NDArray,
        key_type: KEYTYPE = "int",
        spin_name: str | None = None,
    ):
        num_spin = len(model)
        model_dict = {}
        # Linear & quadratic term
        for idx_i in range(num_spin):
            for idx_j in range(idx_i, num_spin):
                coef = model[idx_i, idx_j]
                if coef != 0:
                    if key_type == "int":
                        model_dict[(idx_i, idx_j)] = coef
                    elif key_type == "str":
                        model_dict[f"{spin_name}{idx_i}, {spin_name}{idx_j}"] = coef
        return model_dict

    def _to_dict_qd_ln(
        self,
        model: npt.NDArray,
        key_type: KEYTYPE = "int",
        spin_name: str | None = None,
    ):
        num_spin = len(model)
        model_dict_qd = {}
        model_dict_ln = {}
        # Linear term
        for idx_i in range(num_spin):
            if key_type == "int":
                model_dict_ln[idx_i] = model[idx_i, idx_i]
            elif key_type == "str":
                model_dict_ln[f"{spin_name}{idx_i}"] = model[idx_i, idx_i]
            # Quadratic term
            for idx_j in range(idx_i + 1, num_spin):
                coef = model[idx_i, idx_j]
                if coef != 0:
                    if key_type == "int":
                        model_dict_qd[(idx_i, idx_j)] = coef
                    elif key_type == "str":
                        model_dict_qd[f"{spin_name}{idx_i}, {spin_name}{idx_j}"] = coef
        return model_dict_ln, model_dict_qd

    def mtx_to_bqm(
        self, model_type: Literal["ISING", "QUBO"], model: npt.NDArray, const: float = 0.0
    ) -> dimod.BinaryQuadraticModel:
        linear, quadratic = self.mtx_to_dict(model=model, union=False, key_type="int")

        if model_type == "ISING":
            vartype = dimod.Vartype.SPIN
        elif model_type == "QUBO":
            vartype = dimod.Vartype.BINARY
        else:
            raise ValueError("model_type must be 'ising' or 'qubo'")

        return dimod.BinaryQuadraticModel(linear, quadratic, const, vartype)

    def show(self, model: npt.NDArray) -> None:
        """Show the model as a heatmap

        Args:
            model (npt.NDArray): Ising/QUBO model
        """
        fig = px.imshow(model)
        fig.show()

    def ising_to_qubo(self, ising: npt.NDArray, const: float = 0.0) -> tuple[npt.NDArray, float]:
        """Convert Ising model to QUBO model

        Args:
            ising (npt.NDArray): Ising model
            const (float): Constant value. Defaults to 0.0.

        Returns:
            tuple[npt.NDArray, float]: QUBO model and constant value
        """
        num_spin = ising.shape[0]
        qubo = np.zeros((num_spin, num_spin), dtype=int)

        for i in range(num_spin):
            for j in range(i + 1, num_spin):
                qubo[i, j] = 4 * ising[i, j]
            qubo[i, i] = 2 * ising[i, i] - 4 * (
                ising[i, :].sum() + ising[:, i].sum() - 2 * qubo[i, i]
            )
        return qubo, ising.sum() - 2 * (np.diag(ising).sum()) + const

    def qubo_to_ising(self, qubo: npt.NDArray, const: float = 0.0) -> tuple[npt.NDArray, float]:
        """Convert QUBO model to Ising model

        Args:
            qubo (npt.NDArray): QUBO model
            const (float, optional): Constant value. Defaults to 0.0.

        Returns:
            tuple[npt.NDArray, float]: Ising model and constant value
        """
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
