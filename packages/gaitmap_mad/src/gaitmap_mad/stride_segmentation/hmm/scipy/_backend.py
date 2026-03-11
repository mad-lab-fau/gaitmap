"""SciPy inference backend."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from gaitmap_mad.stride_segmentation.hmm._backend_base import BaseHmmBackend, BaseTrainableHmm
from gaitmap_mad.stride_segmentation.hmm._backend_common import prepare_predict_data
from gaitmap_mad.stride_segmentation.hmm._config import CompositeHmmConfig, HmmSubModelConfig
from gaitmap_mad.stride_segmentation.hmm._state import HMMState
from gaitmap_mad.stride_segmentation.hmm.scipy._utils import (
    log_emission_probabilities,
    map_decode,
    viterbi_decode,
)


class ScipyHmmInferenceBackend(BaseHmmBackend):
    """SciPy-based inference-only backend operating directly on `HMMState`."""

    def __init__(self, backend_id: str = "scipy-inference") -> None:
        super().__init__(backend_id=backend_id)

    def create_submodel(self, config: HmmSubModelConfig) -> BaseTrainableHmm:
        raise NotImplementedError("ScipyHmmInferenceBackend is inference-only and can not create trainable submodels.")

    def finalize_model(
        self,
        *,
        trained_models: dict[str, BaseTrainableHmm],
        labels_train_sequence: list[np.ndarray],
        data_sequence_feature_space: list[pd.DataFrame],
        data_columns: tuple[str, ...],
        model_config: CompositeHmmConfig,
        module_offsets: dict[str, int],
        initialization: Literal["labels", "fully-connected"],
        algo_train: Literal["viterbi", "baum-welch"],
        stop_threshold: float,
        max_iterations: int,
        verbose: bool,
        n_jobs: int,
        name: str,
    ) -> tuple[HMMState, Any]:
        raise NotImplementedError("ScipyHmmInferenceBackend is inference-only and can not finalize/train models.")

    def predict(
        self,
        model: HMMState,
        data: pd.DataFrame,
        *,
        expected_columns: tuple[str, ...],
        algorithm: Literal["viterbi", "map"],
        verbose: bool,
    ) -> np.ndarray:
        del verbose
        observations = prepare_predict_data(data, expected_columns, len(model.compiled.state_names))
        log_emissions = log_emission_probabilities(model, observations)
        if algorithm == "viterbi":
            return viterbi_decode(model, log_emissions)
        return map_decode(model, log_emissions)


__all__ = ["ScipyHmmInferenceBackend"]
