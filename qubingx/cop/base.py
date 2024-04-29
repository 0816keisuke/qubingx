from abc import ABCMeta, abstractmethod
from enum import Enum, unique


@unique
class Model(Enum):
    ISING = "ISING"
    QUBO = "QUBO"


@unique
class Matrix(Enum):
    upper = "upper"
    lower = "lower"
    sym = "symmetric"


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


class Base(metaclass=ABCMeta):
    @abstractmethod
    def h_all(self):
        pass

    @abstractmethod
    def h_obj(self):
        pass

    @abstractmethod
    def h_pen(self):
        pass

    @abstractmethod
    def energy(self):
        pass

    @abstractmethod
    def check_constraint(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass

    @abstractmethod
    def to_bqm(self):
        pass
