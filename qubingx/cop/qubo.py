from typing import Literal

import numpy as np
from dimod import BinaryQuadraticModel

from qubingx.cop.base import Base, Group, Matrix, Model
from qubingx.cop.errors import Errors
from qubingx.util.util import qubo_to_ising
from qubingx.util.util import to_bqm as util_to_bqm
from qubingx.util.util import to_dict as util_to_dict

group = Literal["all", "obj", "constraint"]
key_type = Literal["int", "str"]


class QUBO(Base):
    def __init__(
        self,
        MODEL: Model,
        MATRIX: Matrix,
        num_spin: int,
        q_all: np.ndarray,
        q_obj: np.ndarray,
        q_constraint: np.ndarray,
        const_all: float,
        const_obj: float,
        const_constraint: float,
    ):
        self.MODEL = Model(MODEL)
        self.MATRIX = Matrix(MATRIX)
        self.num_spin = num_spin
        self.q_all = q_all
        self.q_obj = q_obj
        self.q_constraint = q_constraint
        self.const_all = const_all
        self.const_obj = const_obj
        self.const_constraint = const_constraint

    def h_all(self) -> None:
        self.q_all = self.q_obj + self.q_constraint
        self.const_all = self.const_obj + self.const_constraint

    def h_obj(self) -> None:
        pass

    def h_constraint(self) -> None:
        pass

    def energy(self, x: np.ndarray, group: group = "total") -> float:
        """
        _summary_

        Args:
            x (np.ndarray): _description_
            model (np.ndarray): _description_
            const (float): _description_

        Returns:
            float: _description_
        """
        model = self.q_all
        const = self.const_all
        if group == Group.all.value:
            pass
        elif group == Group.obj.value:
            model = self.q_obj
            const = self.const_obj
        elif group == Group.constraint.value:
            model = self.q_constraint
            const = self.const_constraint
        else:
            raise Errors.GroupError(f"Invalid group name {group}")

        return float(np.dot(np.dot(x, model), x) + const)

    def energy(self, x: np.ndarray, group: group = "total") -> float:
        """
        _summary_

        Args:
            x (np.ndarray): _description_
            model (np.ndarray): _description_
            const (float): _description_

        Returns:
            float: _description_
        """
        group_dict = {
            Group.all_.value: (self.q_all, self.const_all),
            Group.obj.value: (self.q_obj, self.const_obj),
            Group.constraint.value: (self.q_constraint, self.const_constraint),
        }

        if group not in group_dict:
            raise Errors.GroupError(f"Invalid group name {group}")

        model, const = group_dict[group]

        return float(np.dot(np.dot(x, model), x) + const)

    def check_constraint(self, x: np.ndarray) -> bool:
        """
        Check if the solution satisfies the constraint.

        Args:
            x (np.ndarray): a set of binary variables you want to check.

        Returns:
            bool: Return True if the solution satisfies the constraint.
                  Return False if the solution does not satisfy the constraint.
        """
        energy = self.energy(x=x, group="constraint")
        return True if energy == 0.0 else False

    def to_dict(
        self,
        group: group = "total",
        union: bool = True,
        key_type: key_type = "int",
        spin_name: str = "",
    ):
        """
        _summary_

        Args:
            group (str, optional): _description_. Defaults to "total".
            union (bool, optional): _description_. Defaults to True.
            key_type (str, optional): _description_. Defaults to "int".
            spin_name (str, optional): _description_. Defaults to "".

        Raises:
            KeyError: _description_

        Returns:
            _type_: _description_
        """
        group_dict = {
            Group.all_.value: self.q_all,
            Group.obj.value: self.q_obj,
            Group.constraint.value: self.q_constraint,
        }

        if group not in group_dict:
            raise Errors.GroupError(f"Invalid group name {group}")

        model_mtx = group_dict[group]

        return util_to_dict(
            model_mtx=model_mtx, union=union, key_type=key_type, spin_name=spin_name
        )

    def to_bqm(self, group: group = "total") -> BinaryQuadraticModel:
        """
        _summary_

        Args:
            group (str, optional): _description_. Defaults to "total".

        Raises:
            KeyError: _description_

        Returns:
            _type_: _description_
        """
        group_dict = {
            Group.all_.value: (self.q_all, self.const_all),
            Group.obj.value: (self.q_obj, self.const_obj),
            Group.constraint.value: (self.q_constraint, self.const_constraint),
        }

        if group not in group_dict:
            raise Errors.GroupError(f"Invalid group name {group}")

        model_mtx, model_const = group_dict[group]

        return util_to_bqm(
            model_mtx=model_mtx, MODEL=self.MODEL.value, const=model_const
        )

    def show(self, group: str = None) -> None:
        """
        _summary_

        Args:
            group (str, optional): _description_. Defaults to None.

        Raises:
            KeyError: _description_
        """
        import plotly.express as px

        model_mtx = self.q_all
        if group == Group.all.value:
            pass
        elif group == Group.obj.value:
            model_mtx = self.q_obj
        elif group == Group.constraint.value:
            model_mtx = self.q_constraint
        else:
            raise Errors.GroupError(f"Invalid group name {group}")
        fig = px.imshow(model_mtx)
        fig.show()

    def to_ising(self):
        """
        _summary_

        Returns:
            _type_: _description_
        """
        from qubingx.cop.ising import ISING

        return ISING(
            MODEL="ISING",
            MATRIX="upper",
            num_spin=self.num_spin,
            j_all=qubo_to_ising(self.q_all, self.const_all)[0],
            j_obj=qubo_to_ising(self.q_obj, self.const_obj)[0],
            j_constraint=qubo_to_ising(self.q_constraint, self.const_constraint)[0],
            const_all=qubo_to_ising(self.q_all, self.const_all)[1],
            const_obj=qubo_to_ising(self.q_obj, self.const_obj)[1],
            const_constraint=qubo_to_ising(self.q_constraint, self.const_constraint)[1],
        )
