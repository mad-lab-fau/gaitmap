"""Legacy pomegranate backend."""

from __future__ import annotations

import copy
import warnings
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import pomegranate as pg
from tpcp import OptiPara, make_optimize_safe
from typing_extensions import Self

from gaitmap_mad.stride_segmentation.hmm._backend_base import BaseHmmBackend, BaseTrainableHmm
from gaitmap_mad.stride_segmentation.hmm._backend_common import (
    extract_cross_module_transitions,
    normalize_transition_and_end_probs,
)
from gaitmap_mad.stride_segmentation.hmm._config import CompositeHmmConfig, HmmSubModelConfig
from gaitmap_mad.stride_segmentation.hmm._state import BackendInfo, HMMState, HmmSubModelState
from gaitmap_mad.stride_segmentation.hmm._utils import (
    create_state_names,
    create_transition_matrix_fully_connected,
    create_transition_matrix_left_right,
    estimate_sequence_boundary_probs,
    extract_transitions_starts_stops_from_hidden_state_sequence,
)
from gaitmap_mad.stride_segmentation.hmm.legacy._state import (
    hmm_state_to_pomegranate_model,
    pomegranate_model_to_flat_hmm_state,
    pomegranate_model_to_hmm_state,
)
from gaitmap_mad.stride_segmentation.hmm.legacy._utils import (
    History,
    ShortenedHMMPrint,
    _clone_model,
    _HackyClonableHMMFix,
    check_history_for_training_failure,
    fix_model_names,
    get_model_distributions,
    gmms_from_samples,
    labels_to_strings,
    predict,
)


def _get_pomegranate_version() -> str | None:
    try:
        return version("pomegranate")
    except PackageNotFoundError:
        return None


def _require_legacy_pomegranate():
    legacy_hmm = getattr(pg, "HiddenMarkovModel", None)
    if legacy_hmm is None:
        raise ImportError("The legacy HMM backend requires pomegranate 0.x with `HiddenMarkovModel` support.")
    return pg


def initialize_hmm(
    data_train_sequence: list[np.ndarray],
    labels_initialization_sequence: list[np.ndarray],
    *,
    n_states: int,
    n_gmm_components: int,
    architecture: Literal["left-right-strict", "left-right-loose", "fully-connected"],
    name: str = "untrained",
    verbose: bool = False,
) -> Any:
    """Initialize a legacy pomegranate HMM from labeled clusters."""
    if architecture not in ["left-right-strict", "left-right-loose", "fully-connected"]:
        raise ValueError(
            'Invalid architecture given. Must be either "left-right-strict", "left-right-loose" or "fully-connected"'
        )

    distributions, _ = gmms_from_samples(
        data_train_sequence,
        labels_initialization_sequence,
        n_gmm_components,
        n_states,
        verbose=verbose,
    )

    if architecture == "left-right-strict":
        transition_matrix, start_probs, end_probs = create_transition_matrix_left_right(n_states, self_transition=False)
    elif architecture == "left-right-loose":
        transition_matrix, _, _ = create_transition_matrix_left_right(n_states, self_transition=True)
        start_probs = np.ones(n_states, dtype=float)
        end_probs = np.ones(n_states, dtype=float)
    else:
        transition_matrix, start_probs, end_probs = create_transition_matrix_fully_connected(n_states)

    pg = _require_legacy_pomegranate()
    model = pg.HiddenMarkovModel.from_matrix(
        transition_probabilities=transition_matrix,
        distributions=copy.deepcopy(distributions),
        starts=start_probs,
        ends=end_probs,
        verbose=verbose,
    )
    model = fix_model_names(model)
    model.bake()
    model.name = name
    return model


class _LegacyTrainableHmm(BaseTrainableHmm, _HackyClonableHMMFix, ShortenedHMMPrint):
    """Internal wrapper used to train a legacy flat HMM.

    This is a thin wrapper around the legacy `pomegranate.HiddenMarkovModel` class and delegates training and
    inference to that runtime.

    Parameters
    ----------
    n_states
        The number of hidden states in the model.
    n_gmm_components
        The number of Gaussian-mixture components per state.
    architecture
        The HMM topology. Supported values are `"left-right-strict"`, `"left-right-loose"`, and
        `"fully-connected"`.
    algo_train
        Training algorithm used by legacy `pomegranate`.
    stop_threshold
        Training convergence threshold.
    max_iterations
        Maximum number of training iterations.
    verbose
        Whether training progress should be printed.
    n_jobs
        Number of parallel jobs used by legacy `pomegranate`.
    name
        Name assigned to the runtime model.
    model
        Optional pre-existing runtime model.
    data_columns
        Expected feature-space column order used for prediction.
    """

    n_states: int
    n_gmm_components: int
    architecture: Literal["left-right-strict", "left-right-loose", "fully-connected"]
    algo_train: Literal["viterbi", "baum-welch"]
    stop_threshold: float
    max_iterations: int
    verbose: bool
    n_jobs: int
    name: Optional[str]  # noqa: UP045
    model: OptiPara[Optional[Any]]  # noqa: UP045
    data_columns: OptiPara[Optional[tuple[str, ...]]]  # noqa: UP045

    def __init__(
        self,
        n_states: int,
        n_gmm_components: int,
        *,
        architecture: Literal["left-right-strict", "left-right-loose", "fully-connected"] = "left-right-strict",
        algo_train: Literal["viterbi", "baum-welch", "labeled"] = "viterbi",
        stop_threshold: float = 1e-9,
        max_iterations: int = 1e8,
        verbose: bool = True,
        n_jobs: int = 1,
        name: str = "my_model",
        model: Optional[Any] = None,  # noqa: UP045
        data_columns: Optional[tuple[str, ...]] = None,  # noqa: UP045
    ) -> None:
        self.n_states = n_states
        self.n_gmm_components = n_gmm_components
        self.algo_train = algo_train
        self.stop_threshold = stop_threshold
        self.max_iterations = max_iterations
        self.architecture = architecture
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.name = name
        self.model = model
        self.data_columns = data_columns

    def predict_hidden_state_sequence(
        self, feature_data: pd.DataFrame, algorithm: Literal["viterbi", "map"] = "viterbi"
    ) -> np.ndarray:
        return predict(self.model, feature_data, expected_columns=self.data_columns, algorithm=algorithm)

    @make_optimize_safe
    def self_optimize(
        self,
        data_sequence: list[pd.DataFrame],
        labels_sequence: list[np.ndarray | pd.Series | pd.DataFrame],
    ) -> Self:
        return self.self_optimize_with_info(data_sequence, labels_sequence)[0]

    def self_optimize_with_info(
        self,
        data_sequence: list[pd.DataFrame],
        labels_sequence: list[np.ndarray | pd.Series | pd.DataFrame],
    ) -> tuple[Self, History]:
        if len(data_sequence) != len(labels_sequence):
            raise ValueError(
                "The given training sequence and initial training labels do not match in their number of individual "
                f"sequences! len(data_train_sequence_list) = {len(data_sequence)} !=  {len(labels_sequence)} = len("
                "initial_hidden_states_sequence_list)"
            )

        for i, (data, labels) in enumerate(zip(data_sequence, labels_sequence)):
            if len(data) < self.n_states:
                raise ValueError(
                    "Invalid training sequence! At least one training sequence has less samples than the specified "
                    "value of states! "
                    f"For sequence {i}: n_states = {self.n_states} > {len(data)} = len(data)"
                )
            if labels is not None:
                if len(data) != len(labels):
                    raise ValueError(
                        "Invalid training sequence! At least one training sequence has a different number of samples "
                        "than the corresponding label sequence! "
                        f"For sequence {i}: len(data) = {len(data)} != {len(labels)} = len(labels)"
                    )
                if not np.all(np.logical_and(labels >= 0, labels < self.n_states)):
                    raise ValueError(
                        "Invalid label sequence! At least one training sequence contains invalid state labels! "
                        f"For sequence {i}: labels not in [0, {self.n_states})"
                    )

        self.data_columns = tuple(data_sequence[0].columns)
        data_sequence_train = [
            np.ascontiguousarray(dataset[list(self.data_columns)].to_numpy().copy().squeeze())
            for dataset in data_sequence
        ]
        labels_sequence_train = []
        for labels in labels_sequence:
            if labels is None:
                labels_sequence_train.append(None)
                continue
            labels = labels.to_numpy().squeeze() if isinstance(labels, (pd.Series, pd.DataFrame)) else labels.squeeze()
            labels_sequence_train.append(np.ascontiguousarray(labels.copy()))

        if self.model is not None:
            warnings.warn("Model already exists. Overwriting existing model.")

        model_untrained = initialize_hmm(
            data_sequence_train,
            labels_sequence_train,
            n_states=self.n_states,
            n_gmm_components=self.n_gmm_components,
            architecture=self.architecture,
            name=self.name + "-untrained",
        )
        model_trained = _clone_model(model_untrained, assert_correct=False)

        _, history = model_trained.fit(
            sequences=np.array(data_sequence_train, dtype=object),
            labels=np.array(labels_sequence_train, dtype=object),
            algorithm=self.algo_train,
            stop_threshold=self.stop_threshold,
            max_iterations=self.max_iterations,
            return_history=True,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            multiple_check_input=False,
        )
        check_history_for_training_failure(history)
        model_trained.name = self.name + "_trained"
        self.model = model_trained
        return self, history


def _create_trainable_hmm_from_config(config: HmmSubModelConfig) -> _LegacyTrainableHmm:
    return _LegacyTrainableHmm(
        n_states=config.n_states,
        n_gmm_components=config.n_gmm_components,
        architecture=config.architecture,
        algo_train=config.algo_train,
        stop_threshold=config.stop_threshold,
        max_iterations=config.max_iterations,
        verbose=config.verbose,
        n_jobs=config.n_jobs,
        name=config.name,
    )


class PomegranateLegacyHmmBackend(BaseHmmBackend):
    """`pomegranate 0.x` backend for HMM training and inference."""

    def __init__(self, backend_id: str = "pomegranate-legacy") -> None:
        if getattr(pg, "HiddenMarkovModel", None) is None:
            raise ImportError(
                "Failed to initialize `PomegranateLegacyHmmBackend`. "
                "This backend requires `pomegranate 0.x` with `HiddenMarkovModel` support. "
                f"Installed version: {_get_pomegranate_version() or 'not installed'}."
            )
        super().__init__(backend_id=backend_id)

    def create_submodel(self, config: HmmSubModelConfig) -> BaseTrainableHmm:
        return _create_trainable_hmm_from_config(config)

    def predict(
        self,
        model: HMMState,
        data: pd.DataFrame,
        *,
        expected_columns: tuple[str, ...],
        algorithm: Literal["viterbi", "map"],
        verbose: bool,
    ) -> np.ndarray:
        runtime_model = hmm_state_to_pomegranate_model(model, verbose=verbose)
        return predict(runtime_model, data, expected_columns=expected_columns, algorithm=algorithm)

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
        distributions = []
        for module in model_config.modules:
            distributions.extend(get_model_distributions(trained_models[module.name].model))

        model = self._create_combined_model(
            trained_models=trained_models,
            labels_train_sequence=labels_train_sequence,
            distributions=distributions,
            model_config=model_config,
            module_offsets=module_offsets,
            initialization=initialization,
            verbose=verbose,
        )

        labels_train_sequence_str = labels_to_strings(labels_train_sequence)
        data_train_sequence = [
            np.ascontiguousarray(feature_data[list(data_columns)].to_numpy().copy())
            for feature_data in data_sequence_feature_space
        ]

        _, history = model.fit(
            sequences=np.array(data_train_sequence, dtype=object),
            labels=np.array(labels_train_sequence_str, dtype=object).copy(),
            algorithm=algo_train,
            stop_threshold=stop_threshold,
            max_iterations=max_iterations,
            return_history=True,
            verbose=verbose,
            n_jobs=n_jobs,
            multiple_check_input=False,
        )
        check_history_for_training_failure(history)
        model.name = name

        submodel_states = tuple(
            HmmSubModelState(
                name=module.name,
                role=module.role,
                model=pomegranate_model_to_flat_hmm_state(trained_models[module.name].model),
            )
            for module in model_config.modules
        )
        model_state = pomegranate_model_to_hmm_state(
            model,
            submodels=submodel_states,
            backend_info=BackendInfo(backend_id=self.backend_id),
        )
        model_state.cross_module_transitions = extract_cross_module_transitions(model_state, module_offsets)
        return model_state, history

    def _create_combined_model(
        self,
        *,
        trained_models: dict[str, BaseTrainableHmm],
        labels_train_sequence: list[np.ndarray],
        distributions: list[Any],
        model_config: CompositeHmmConfig,
        module_offsets: dict[str, int],
        initialization: Literal["labels", "fully-connected"],
        verbose: bool,
    ) -> Any:
        pg = _require_legacy_pomegranate()

        n_states = sum(module.n_states for module in model_config.modules)
        if initialization == "fully-connected":
            trans_mat, start_probs, end_probs = create_transition_matrix_fully_connected(n_states)
            trans_mat, end_probs = normalize_transition_and_end_probs(trans_mat, end_probs)
            model = pg.HiddenMarkovModel.from_matrix(
                transition_probabilities=copy.deepcopy(trans_mat),
                distributions=copy.deepcopy(distributions),
                starts=start_probs,
                ends=end_probs,
                state_names=list(create_state_names(n_states)),
                verbose=verbose,
            )
        else:
            trans_mat = np.zeros((n_states, n_states))
            for module in model_config.modules:
                module_transition_matrix = trained_models[module.name].model.dense_transition_matrix()[:-2, :-2]
                offset = module_offsets[module.name]
                trans_mat[offset : offset + module.n_states, offset : offset + module.n_states] = (
                    module_transition_matrix
                )

            transitions, _, _ = extract_transitions_starts_stops_from_hidden_state_sequence(labels_train_sequence)
            start_probs, end_probs = estimate_sequence_boundary_probs(labels_train_sequence, n_states)
            for from_state, to_state in transitions:
                trans_mat[int(from_state[1:]), int(to_state[1:])] = max(
                    trans_mat[int(from_state[1:]), int(to_state[1:])],
                    0.1,
                )
            trans_mat, end_probs = normalize_transition_and_end_probs(trans_mat, end_probs)

            model = pg.HiddenMarkovModel.from_matrix(
                transition_probabilities=copy.deepcopy(trans_mat),
                distributions=copy.deepcopy(distributions),
                starts=start_probs,
                ends=end_probs,
                state_names=list(create_state_names(n_states)),
                verbose=verbose,
            )

        model = fix_model_names(model)
        model.bake()
        model.freeze_distributions()
        return _clone_model(model, assert_correct=False)
