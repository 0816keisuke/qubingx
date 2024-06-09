import random
from typing import Literal

import numpy as np
import numpy.typing as npt

MODELTYPE = Literal["ISING", "QUBO"]
MATRIXTYPE = Literal["upper", "lower", "symmetric"]


class RandomGenerator:
    def __init__(self) -> None:
        pass

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

    def random_distance_matrix(
        self, n: int, w_min: int = 1, w_max: int = 9, prob: float = 0, seed: int | None = None
    ) -> npt.NDArray:
        """Generate random distance matrix with n nodes and edge weight range [w_min, w_max]

        Args:
            n (int): Number of nodes
            w_min (int, optional): Minimum weight. Defaults to 1.
            w_max (int, optional): Maximum weight. Defaults to 9.
            prob (float, optional): Probability of non-zero value. Defaults to 0.
            seed (int | None, optional): Random seed. Defaults to None.

        Returns:
            npt.NDArray: Random distance matrix
        """
        if prob < 0 or prob > 1:
            raise ValueError("prob must be in the range [0, 1]")
        if seed is not None:
            random.seed(seed)

        distance_mtx = np.random.randint(w_min, w_max + 1, (n, n))
        if prob > 0:
            distance_mtx[np.random.rand(n, n) < prob] = 0
        # 対角成分を0にする
        np.fill_diagonal(distance_mtx, 0)
        return distance_mtx

    def random_graph(
        self,
        n: int,
        p: float,
        w_min: int = 1,
        w_max: int = 9,
        return_type: Literal["list", "matrix"] = "list",
        seed: int | None = None,
    ) -> npt.NDArray:
        """Generate random graph with n nodes and edge probability p and edge weight range [w_min, w_max]

        Args:
            n (int): Number of nodes
            p (float): Edge density
            seed (int | None, optional): Random seed. Defaults to None.

        Returns:
            npt.NDArray: Adjacency matrix of the random graph
        """
        if p < 0 or p > 1:
            raise ValueError("p must be in the range [0, 1]")
        if seed is not None:
            random.seed(seed)

        num_edge = int(n * (n - 1) / 2 * p)
        # 完全グラフの辺のリスト
        all_edge_idx = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]
        # all_edge_idxからどれを重み0ではない辺とするか
        act_edge_idx = sorted(random.sample(list(range(int(n * (n - 1) / 2))), num_edge))
        # 選ばれた辺に対する重み
        act_weights = [
            random.choice([i for i in range(w_min, w_max + 1) if i != 0]) for _ in range(num_edge)
        ]

        # 選ばれた辺と重みのリスト
        edge_weights = [[0] * 3] * num_edge
        for i, idx in enumerate(act_edge_idx):
            edge_weights[i] = [all_edge_idx[idx][0], all_edge_idx[idx][1], act_weights[i]]

        if return_type == "list":
            return edge_weights
        elif return_type == "matrix":
            adjacency_mtx = np.zeros((n, n), dtype=int)
            for i, j, w in edge_weights:
                adjacency_mtx[i][j] = w
                adjacency_mtx[j][i] = w
            return adjacency_mtx
        else:
            raise ValueError("return_type must be 'list' or 'matrix'")

    def random_model(
        self,
        n: int,
        v_min: int = -5,
        v_max: int = 5,
        prob: float = 0,
        model_type: MATRIXTYPE = "upper",
        seed: int | None = None,
    ) -> npt.NDArray:
        """Generate random Ising/QUBO model with n spins, value range [v_min, v_max], and probability prob

        Args:
            n (int): Number of spins
            v_min (int, optional): Minimum value of the model. Defaults to -5.
            v_max (int, optional): Maximum value of the model. Defaults to 5.
            prob (float, optional): Probability of non-zero value. Defaults to 0.
            model_type (Literal[&quot;upper&quot;, &quot;lower&quot;, &quot;sym&quot;]):
                Model type. Defaults to &quot;upper&quot;.
                "upper": Upper triangular matrix
                "lower": Lower triangular matrix
                "symmetric": Symmetric matrix
            seed (int | None, optional): Random seed. Defaults to None.

        Returns:
            npt.NDArray: Random Ising/QUBO model
        """
        if seed is not None:
            np.random.seed(seed)

        model = np.random.randint(v_min, v_max + 1, (n, n))
        if prob > 0:
            model[np.random.rand(n, n) < prob] = 0
        model = np.triu(model, k=0)

        if model_type == "upper":
            return model
        elif model_type == "lower":
            return model.T
        elif model_type == "symmetric":
            return (model + model.T) / 2
        else:
            raise ValueError("model_type must be 'upper', 'lower' or 'sym'")

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

    def random_twoway_onehot_solution(
        self, n: int, mtx: bool = False, seed: int | None = None
    ) -> list[int]:
        """Generate random QUBO model solution which satisfy 2-way-1hot constraint

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
