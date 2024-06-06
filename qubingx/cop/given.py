import numpy as np
import numpy.typing as npt

from qubingx.cop.base import Base, Model


class Given(Base):
    def __init__(
        self,
        mtx_all: list[list[float | int]] | npt.NDArray,
        mtx_obj: list[list[float | int]] | npt.NDArray | None = None,
        mtx_constraint: list[list[float | int]] | npt.NDArray | None = None,
        offset_all: float = 0 ,
        offset_obj: float = 0,
        offset_constraint: float = 0,
    ):

        num_spin = len(mtx_all)
        super().__init__(
            model=Model("QUBO"),
            num_spin=num_spin,
            h_all=mtx_all,
            h_obj=mtx_obj if mtx_obj is not None else np.zeros((num_spin, num_spin)),
            h_constraint=mtx_constraint if mtx_constraint is not None else np.zeros((num_spin, num_spin)),
            const_all=offset_all,
            const_obj=offset_obj,
            const_constraint=offset_constraint,
        )

        self._make_h_obj()
        self._make_h_constraint()
        # self._make_h_all()

    def _make_h_obj(self):
        pass

    def _make_h_constraint(self):
        pass
