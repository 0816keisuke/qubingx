from typing import Any, Literal

import numpy as np
from dimod import BinaryQuadraticModel
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler


# NOTE: Builderパターンに変更したほうが良い？
class DWaveAnnealParams:
    # https://docs.dwavesys.com/docs/latest/c_solver_parameters.html
    def __init__(
        self,
        anneal_offsets: list[int] | None = None,
        anneal_schedule: list[list[float]] = None,
        annealing_time: int | float | None = None,
        answer_mode: Literal["Raw", "histogram"] = "histogram",
        auto_scale: bool | None = None,
        fast_anneal: bool | None = None,
        flux_biases: list[int] | None = None,
        flux_drift_compensation: bool = True,
        h_gain_schedule: list[list[float]] | None = None,
        initial_state: Any | None = None,  # ほんとはdict[int, int]
        max_answers: int | None = None,
        num_reads: int = 1,
        programming_thermalization: float | None = None,
        readout_thermalization: float | None = None,
        reduce_intersample_correlation: bool = False,
        reinitialize_state: bool | None = None,
        chain_strength: Any | None = None,
        chain_break_method: Any | None = None,
        chain_break_fraction: bool | None = None,
        embedding_parameters: Any | None = None,
        return_embedding: bool | None = None,
    ):
        if anneal_offsets is not None:
            self.anneal_offsets = anneal_offsets
        if anneal_schedule is not None:
            self.anneal_schedule = anneal_schedule
        if annealing_time is not None:
            self.annealing_time = annealing_time
        self.answer_mode = answer_mode
        if auto_scale is not None:
            self.auto_scale = auto_scale
        if fast_anneal is not None:
            self.fast_anneal = fast_anneal
        if flux_biases is not None:
            self.flux_biases = flux_biases
        self.flux_drift_compensation = flux_drift_compensation
        if h_gain_schedule is not None:
            self.h_gain_schedule = h_gain_schedule
        if initial_state is not None:
            self.initial_state = initial_state
        if max_answers is not None:
            self.max_answers = max_answers
        self.num_reads = num_reads
        if programming_thermalization is not None:
            self.programming_thermalization = programming_thermalization
        if readout_thermalization is not None:
            self.readout_thermalization = readout_thermalization
        self.reduce_intersample_correlation = reduce_intersample_correlation
        if reinitialize_state is not None:
            self.reinitialize_state = reinitialize_state
        if chain_strength is not None:
            self.chain_strength = chain_strength
        if chain_break_method is not None:
            self.chain_break_method = chain_break_method
        if chain_break_fraction is not None:
            self.chain_break_fraction = chain_break_fraction
        if embedding_parameters is not None:
            self.embedding_parameters = embedding_parameters
        if return_embedding is not None:
            self.return_embedding = return_embedding


class DWaveQPU:
    def __init__(self, anneal_params: DWaveAnnealParams, config_file: str | None = None):
        if config_file is None:
            raise ValueError("Configuration file for D-Wave is required.")
        else:
            self.config_file = config_file
        self.anneal_params = anneal_params
        self.sampler = EmbeddingComposite(DWaveSampler(config_file=config_file))

    def sample(self, bqm: BinaryQuadraticModel, raw: bool = False):
        result = self.sampler.sample(bqm, **self.anneal_params.__dict__)
        result.record.sort(order="energy")

        energies = []
        solutions = []
        for i, num_occ in enumerate(result.record.num_occurrences):
            # print(i, num_occ)
            for _ in range(num_occ):
                energies.append(result.record.energy[i])
                solutions.append(result.record.sample[i])
        solutions = np.array(solutions)

        if raw:
            return energies, solutions, result
        else:
            return energies, solutions
