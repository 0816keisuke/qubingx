from typing import Any, Literal

import numpy as np
from dimod import BinaryQuadraticModel
from dwave.samplers import SimulatedAnnealingSampler


class SAParams:
    # https://docs.ocean.dwavesys.com/en/latest/docs_samplers/generated/dwave.samplers.SimulatedAnnealingSampler.sample.html
    def __init__(
        self,
        beta_range: list[float] | tuple[float, float] | None = None,
        num_reads: int | None = None,
        num_sweeps: int | None = None,
        num_sweeps_per_beta: int = 1,
        beta_schedule_type: Literal["linear", "geometric", "custom"] = "geometric",
        seed: int | None = None,
        interrupt_function: Any | None = None,
        beta_schedule: Any | None = None,
        initial_states: dict[int, int] | None = None,  # 本当はもっと複雑だけど面倒なので省略
        initial_states_generator: Literal["none", "tile", "random"] = "random",
        randomize_order: bool = False,
        proposal_acceptance_criteria: str = "Metropolis",
    ):
        if beta_range is not None:
            self.beta_range = beta_range
        if num_reads is not None:
            self.num_reads = num_reads
        if num_sweeps is not None:
            self.num_sweeps = num_sweeps
        self.num_sweeps_per_beta = num_sweeps_per_beta
        self.beta_schedule_type = beta_schedule_type
        if seed is not None:
            self.seed = seed
        if interrupt_function is not None:
            self.interrupt_function = interrupt_function
        if beta_schedule is not None:
            self.beta_schedule = beta_schedule
        if initial_states is not None:
            self.initial_states = initial_states
        self.initial_states_generator = initial_states_generator
        self.randomize_order = randomize_order
        self.proposal_acceptance_criteria = proposal_acceptance_criteria


class SA:
    def __init__(self, anneal_params: SAParams):
        self.anneal_params = anneal_params
        self.sampler = SimulatedAnnealingSampler()

    def sample(self, bqm: BinaryQuadraticModel, raw: bool = False):
        result = self.sampler.sample(bqm, **self.anneal_params.__dict__)
        result.record.sort(order="energy")
        energies = result.record.energy
        solutions = []
        for i, num_occ in enumerate(result.record.num_occurrences):
            for _ in range(num_occ):
                solutions.append(result.record.sample[i])
        solutions = np.array(solutions)

        if raw:
            return energies, solutions, result
        else:
            return energies, solutions
