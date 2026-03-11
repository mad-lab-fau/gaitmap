"""Base backend abstractions and backend selection."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Literal, Sequence, TypeVar

import numpy as np
import pandas as pd
from typing_extensions import Self

from gaitmap.base import _BaseSerializable

if TYPE_CHECKING:
    from gaitmap_mad.stride_segmentation.hmm._config import CompositeHmmConfig, HmmSubModelConfig
    from gaitmap_mad.stride_segmentation.hmm._state import HMMState

TrainableModelT = TypeVar("TrainableModelT")
HmmInferenceResult = np.ndarray
HmmTrainingResult = tuple[TrainableModelT, Any]


class BaseTrainableHmm(_BaseSerializable):
    """Backend-specific trainable flat-HMM primitive."""

    def predict_hidden_state_sequence(
        self,
        feature_data: pd.DataFrame,
        algorithm: Literal["viterbi", "map"] = "viterbi",
    ) -> HmmInferenceResult:
        raise NotImplementedError

    def self_optimize_with_info(
        self,
        data_sequence: Sequence[pd.DataFrame | np.ndarray],
        labels_sequence: Sequence[np.ndarray],
    ) -> HmmTrainingResult[Self]:
        raise NotImplementedError


class BaseHmmBackend(_BaseSerializable):
    """Base abstraction for backend-specific HMM primitives."""

    backend_id: str

    def __init__(self, backend_id: str) -> None:
        self.backend_id = backend_id

    def create_submodel(self, config: HmmSubModelConfig) -> BaseTrainableHmm:
        raise NotImplementedError

    def predict(
        self,
        model: HMMState,
        data: pd.DataFrame,
        *,
        expected_columns: tuple[str, ...],
        algorithm: Literal["viterbi", "map"],
        verbose: bool,
    ) -> HmmInferenceResult:
        raise NotImplementedError

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
    ) -> HmmTrainingResult[HMMState]:
        raise NotImplementedError


def get_default_hmm_backend() -> BaseHmmBackend:
    """Return the default runtime backend for the installed environment."""
    try:
        return import_module("gaitmap_mad.stride_segmentation.hmm.modern").PomegranateModernHmmBackend()
    except ImportError:
        pass
    try:
        return import_module("gaitmap_mad.stride_segmentation.hmm.legacy").PomegranateLegacyHmmBackend()
    except ImportError:
        return import_module("gaitmap_mad.stride_segmentation.hmm.scipy").ScipyHmmInferenceBackend()
