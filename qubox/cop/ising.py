from dataclasses import dataclass

import numpy as np

from qubox.cop.base import Base, Group, Matrix, Model
from qubox.cop.errors import Errors
from qubox.util.util import ising_to_qubo
from qubox.util.util import to_bqm as util_to_bqm
from qubox.util.util import to_dict as util_to_dict


@dataclass
class ISING(Base):
    """
    _summary_

    Args:
        Base (_type_): _description_

    Raises:
        KeyError: _description_
        KeyError: _description_
        KeyError: _description_

    Returns:
        _type_: _description_
    """

    MODEL: Model
    MATRIX: Matrix

    num_spin: int

    j_all: np.ndarray
    j_obj: np.ndarray
    j_constraint: np.ndarray

    const_all: float
    const_obj: float
    const_constraint: float

    def h_all(self):
        self.j_all = self.j_obj + self.j_constraint
        self.const_all = self.const_obj + self.const_constraint

    def h_obj(self) -> None:
        pass

    def h_constraint(self) -> None:
        pass

    def energy(self, x: np.ndarray, model: np.ndarray, const: float) -> float:
        """
        _summary_

        Args:
            x (np.ndarray): _description_
            model (np.ndarray): _description_
            const (float): _description_

        Returns:
            float: _description_
        """
        return float(np.dot(np.dot(x, np.triu(model, k=1)), x)) + np.dot(
            x, np.diag(model) + const
        )

    def constraint(self, x: np.ndarray) -> bool:
        """
        _summary_

        Args:
            x (np.ndarray): _description_

        Returns:
            bool: _description_
        """
        energy = self.energy(x=x, model=self.j_constraint, const=self.const_constraint)
        return True if energy == 0.0 else False

    def to_dict(
        self,
        group: str = "all",
        union: bool = True,
        key_type: str = "int",
        spin_name: str = "",
    ):
        """
        _summary_

        Args:
            group (str, optional): _description_. Defaults to "all".
            union (bool, optional): _description_. Defaults to True.
            key_type (str, optional): _description_. Defaults to "int".
            spin_name (str, optional): _description_. Defaults to "".

        Raises:
            KeyError: _description_

        Returns:
            _type_: _description_
        """
        model_mtx = self.q_all
        if group == Group.all.value:
            pass
        elif group == Group.obj.value:
            model_mtx = self.q_obj
        elif group == Group.constraint.value:
            model_mtx = self.q_constraint
        else:
            raise Errors.GroupError(f"Invalid group name {group}")
        return util_to_dict(
            model_mtx, union=union, key_type=key_type, spin_name=spin_name
        )

    def to_bqm(self, group: str = "all"):
        """
        _summary_

        Args:
            group (str, optional): _description_. Defaults to "all".

        Raises:
            KeyError: _description_

        Returns:
            _type_: _description_
        """
        model_mtx = self.q_all
        model_const = self.const_all
        if group == Group.all.value:
            pass
        elif group == Group.obj.value:
            model_mtx = self.q_obj
            model_const = self.const_obj
        elif group == Group.constraint.value:
            model_mtx = self.q_constraint
            model_const = self.const_constraint
        else:
            raise Errors.GroupError(f"Invalid group name {group}")
        return util_to_bqm(model_mtx=model_mtx, MODEL=self.MODEL, const=model_const)

    def show(self, group: str = None):
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

    def to_qubo(self):
        """
        _summary_

        Returns:
            _type_: _description_
        """
        from qubox.cop.qubo import QUBO

        return QUBO(
            MODEL="QUBO",
            MATRIX="upper",
            num_spin=self.num_spin,
            q_all=ising_to_qubo(self.j_all, self.const_all)[0],
            q_obj=ising_to_qubo(self.j_obj, self.const_obj)[0],
            q_constraint=ising_to_qubo(self.j_constraint, self.const_constraint)[0],
            const_all=ising_to_qubo(self.j_all, self.const_all)[1],
            const_obj=ising_to_qubo(self.j_obj, self.const_obj)[1],
            const_constraint=ising_to_qubo(self.j_constraint, self.const_constraint)[1],
        )
