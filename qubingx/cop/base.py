from enum import Enum, unique
from typing import Literal

import numpy.typing as npt
from dimod import BinaryQuadraticModel

from qubingx.cop.errors import Errors
from qubingx.tools import QubingUtils

groups = Literal["all", "obj", "constraint"]
key_type = Literal["int", "str"]
utils = QubingUtils()


@unique
class Model(Enum):
    ISING = "ISING"
    QUBO = "QUBO"


@unique
class Group(Enum):
    all_ = "all"
    obj = "obj"
    constraint = "constraint"


@unique
class Encoding(Enum):
    one_hot = "1-hot"
    binary = "binary"
    unary = "unary"


class Base:
    def __init__(
        self,
        model: Model,
        num_spin: int,
        h_all: npt.NDArray,
        h_obj: npt.NDArray,
        h_constraint: npt.NDArray,
        const_all: float,
        const_obj: float,
        const_constraint: float,
    ):
        self.model = Model(model)
        self.num_spin = num_spin
        self.h_all = h_all
        self.h_obj = h_obj
        self.h_constraint = h_constraint
        self.const_all = const_all
        self.const_obj = const_obj
        self.const_constraint = const_constraint

    def _select_group(self, group: groups) -> tuple[npt.NDArray, float]:
        if group == Group.all_.value:
            return self.h_all, self.const_all
        elif group == Group.obj.value:
            return self.h_obj, self.const_obj
        elif group == Group.constraint.value:
            return self.h_constraint, self.const_constraint
        else:
            raise Errors.GroupError(f"Invalid group name {group}")

    def _make_h_all(self) -> None:
        self.h_all = self.h_obj + self.h_constraint
        self.const_all = self.const_obj + self.const_constraint

    def _make_h_obj(self) -> None:
        pass

    def _make_h_constraint(self) -> None:
        pass

    def energy(self, solution: npt.NDArray, group: groups = "all") -> float:
        model, const = self._select_group(group)
        return utils.calc_energy(
            model_type=self.model.value, model=model, solution=solution, const=const
        )

    def check_constraint(self, solution: npt.NDArray) -> bool:
        """
        Check if the solution satisfies the constraint.

        Args:
            solution (npt.NDArray): a set of binary variables you want to check.

        Returns:
            bool: Return True if the solution satisfies the constraint.
                  Return False if the solution does not satisfy the constraint.
        """
        return utils.check_constraint(
            model_type=self.model.value,
            model=self.h_constraint,
            solution=solution,
            const=self.const_constraint,
        )

    def to_dict(
        self,
        group: groups = "all",
        union: bool = True,
        key_type: key_type = "int",
        spin_name: str = "",
    ):
        model, _ = self._select_group(group)
        return utils.mtx_to_dict(model=model, union=union, key_type=key_type, spin_name=spin_name)

    def to_bqm(self, group: groups = "all") -> BinaryQuadraticModel:
        model, const = self._select_group(group)
        return utils.mtx_to_bqm(model_type=self.model.value, model=model, const=const)

    def show(self, group: str = None) -> None:
        model, _ = self._select_group(group)
        utils.show(model=model)

    def to_qubo(self):
        if self.model == Model.ISING:
            self.model = Model.QUBO
            self.h_all, self.const_all = utils.qubo_to_ising(self.h_all, self.const_all)
            self.h_obj, self.const_obj = utils.qubo_to_ising(self.h_obj, self.const_obj)
            self.h_constraint, self.const_constraint = utils.qubo_to_ising(
                self.h_constraint, self.const_constraint
            )
        elif self.model == Model.QUBO:
            raise Errors.ModelError("The model is already QUBO.")
        else:
            raise Errors.ModelError(f"Invalid model name {self.model}")

    def to_ising(self):
        if self.model == Model.ISING:
            raise Errors.ModelError("The model is already ISING.")
        elif self.model == Model.QUBO:
            self.model = Model.ISING
            self.h_all, self.const_all = utils.qubo_to_ising(self.h_all, self.const_all)
            self.h_obj, self.const_obj = utils.qubo_to_ising(self.h_obj, self.const_obj)
            self.h_constraint, self.const_constraint = utils.qubo_to_ising(
                self.h_constraint, self.const_constraint
            )
        else:
            raise Errors.ModelError(f"Invalid model name {self.model}")
