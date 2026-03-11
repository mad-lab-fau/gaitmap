"""Modern pomegranate backend."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal

import numpy as np
import pandas as pd
try:
    from pomegranate.hmm import DenseHMM
except (ImportError, AttributeError):  # pragma: no cover - exercised in environments without modern pomegranate
    DenseHMM = None
try:
    import torch
except ImportError:  # pragma: no cover - exercised in environments without torch
    torch = None

from gaitmap_mad.stride_segmentation.hmm._backend_base import BaseHmmBackend, BaseTrainableHmm
from gaitmap_mad.stride_segmentation.hmm._backend_common import (
    extract_cross_module_transitions,
    normalize_transition_and_end_probs,
    prepare_predict_data,
)
from gaitmap_mad.stride_segmentation.hmm._config import CompositeHmmConfig, HmmSubModelConfig
from gaitmap_mad.stride_segmentation.hmm._state import BackendInfo, FlatHmmState, HmmGraphState, HMMState, HmmSubModelState
from gaitmap_mad.stride_segmentation.hmm._utils import (
    create_state_names,
    create_transition_matrix_fully_connected,
    estimate_sequence_boundary_probs,
    extract_transitions_starts_stops_from_hidden_state_sequence,
)
from gaitmap_mad.stride_segmentation.hmm.modern._state import (
    flat_hmm_state_to_pomegranate_modern_model,
    hmm_state_to_pomegranate_modern_model,
    pomegranate_modern_model_to_flat_hmm_state,
    pomegranate_modern_model_to_hmm_state,
)
from gaitmap_mad.stride_segmentation.hmm.modern._utils import (
    PomegranateModernHistory,
    freeze_emission,
    labels_to_priors,
    to_modern_input,
    to_training_arrays,
    trainable_state_from_clusters,
)


def _get_pomegranate_version() -> str | None:
    try:
        return version("pomegranate")
    except PackageNotFoundError:
        return None


def _predict_with_native_runtime_model(
    runtime_model: Any,
    observations: np.ndarray,
    *,
    algorithm: Literal["viterbi", "map"],
) -> np.ndarray:
    runtime_input = torch.tensor(observations, dtype=torch.float64).unsqueeze(0)
    with torch.no_grad():
        if algorithm == "viterbi":
            return runtime_model.viterbi(runtime_input).detach().cpu().numpy()[0]
        return runtime_model.predict(runtime_input).detach().cpu().numpy()[0]


class PomegranateModernTrainableHmm(BaseTrainableHmm):
    """Internal trainable HMM wrapper used by the modern pomegranate backend."""

    def __init__(self, config: HmmSubModelConfig, backend_id: str) -> None:
        self.config = config
        self.backend_id = backend_id
        self.data_columns: tuple[str, ...] | None = None
        self.model: FlatHmmState | None = None

    def predict_hidden_state_sequence(
        self, feature_data: pd.DataFrame, algorithm: Literal["viterbi", "map"] = "viterbi"
    ) -> np.ndarray:
        if self.model is None or self.data_columns is None:
            raise ValueError("No trained model available. Call `self_optimize` first.")
        observations = prepare_predict_data(feature_data, self.data_columns, len(self.model.state_names))
        runtime_model = flat_hmm_state_to_pomegranate_modern_model(
            self.model,
            verbose=self.config.verbose,
            max_iterations=self.config.max_iterations,
            stop_threshold=self.config.stop_threshold,
        ).to(torch.float64)
        return _predict_with_native_runtime_model(runtime_model, observations, algorithm=algorithm)

    def self_optimize_with_info(
        self,
        data_sequence: list[pd.DataFrame],
        labels_sequence: list[np.ndarray],
    ) -> tuple[PomegranateModernTrainableHmm, PomegranateModernHistory]:
        if self.config.algo_train != "baum-welch":
            raise NotImplementedError(
                "The modern pomegranate backend currently only supports `baum-welch` training."
            )
        if len(data_sequence) != len(labels_sequence):
            raise ValueError(
                "The given training sequence and initial training labels do not match in their number of individual "
                f"sequences! len(data_sequence) = {len(data_sequence)} != {len(labels_sequence)} = len(labels_sequence)"
            )

        self.data_columns = tuple(data_sequence[0].columns)
        arrays = to_training_arrays(data_sequence, data_columns=self.data_columns)
        flat_state = trainable_state_from_clusters(
            arrays,
            labels_sequence,
            n_states=self.config.n_states,
            n_gmm_components=self.config.n_gmm_components,
            architecture=self.config.architecture,
            name=self.config.name,
            normalize_transition_and_end_probs=normalize_transition_and_end_probs,
        )
        runtime_model = flat_hmm_state_to_pomegranate_modern_model(
            flat_state,
            verbose=self.config.verbose,
            max_iterations=self.config.max_iterations,
            stop_threshold=self.config.stop_threshold,
        )
        runtime_model.fit(to_modern_input(arrays))
        self.model = pomegranate_modern_model_to_flat_hmm_state(
            runtime_model,
            state_names=flat_state.state_names,
            name=f"{self.config.name}_trained",
        )
        return self, PomegranateModernHistory()


class PomegranateModernHmmBackend(BaseHmmBackend):
    """`pomegranate 1.x` backend using DenseHMM for training."""

    inference_implementation: Literal["canonical", "native"]

    def __init__(
        self,
        backend_id: str = "pomegranate-modern",
        *,
        inference_implementation: Literal["canonical", "native"] = "native",
    ) -> None:
        if DenseHMM is None:
            raise ImportError(
                "Failed to initialize `PomegranateModernHmmBackend`. "
                "This backend requires `pomegranate 1.x` with `DenseHMM` support. "
                f"Installed version: {_get_pomegranate_version() or 'not installed'}."
            )
        if torch is None:
            raise ImportError(
                "Failed to initialize `PomegranateModernHmmBackend`. "
                "This backend requires `torch` because native `pomegranate 1.x` inference and training run on PyTorch."
            )
        super().__init__(backend_id=backend_id)
        self.inference_implementation = inference_implementation

    def create_submodel(self, config: HmmSubModelConfig) -> BaseTrainableHmm:
        return PomegranateModernTrainableHmm(config, backend_id=self.backend_id)

    def predict(
        self,
        model: HMMState,
        data: pd.DataFrame,
        *,
        expected_columns: tuple[str, ...],
        algorithm: Literal["viterbi", "map"],
        verbose: bool,
    ) -> np.ndarray:
        observations = prepare_predict_data(data, expected_columns, len(model.compiled.state_names))
        return self._predict_native(model, observations, algorithm=algorithm, verbose=verbose)

    def _predict_native(
        self,
        model: HMMState,
        observations: np.ndarray,
        *,
        algorithm: Literal["viterbi", "map"],
        verbose: bool,
    ) -> np.ndarray:
        runtime_model = hmm_state_to_pomegranate_modern_model(model, verbose=verbose).to(torch.float64)
        return _predict_with_native_runtime_model(runtime_model, observations, algorithm=algorithm)

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
        del n_jobs
        if algo_train != "baum-welch":
            raise NotImplementedError("The modern pomegranate backend currently only supports `baum-welch` training.")

        submodel_states = tuple(
            HmmSubModelState(name=module.name, role=module.role, model=trained_models[module.name].model)
            for module in model_config.modules
        )
        combined_state = self._create_combined_state(
            trained_models=trained_models,
            labels_train_sequence=labels_train_sequence,
            model_config=model_config,
            module_offsets=module_offsets,
            initialization=initialization,
            name=name,
        )
        runtime_model = flat_hmm_state_to_pomegranate_modern_model(
            combined_state,
            verbose=verbose,
            max_iterations=max_iterations,
            stop_threshold=stop_threshold,
        )
        priors = labels_to_priors(labels_train_sequence, len(combined_state.state_names))
        training_data = to_modern_input(to_training_arrays(data_sequence_feature_space, data_columns=data_columns))
        runtime_model.fit(training_data, priors=priors)

        model_state = pomegranate_modern_model_to_hmm_state(
            runtime_model,
            submodels=submodel_states,
            backend_info=BackendInfo(backend_id=self.backend_id, backend_version=_get_pomegranate_version()),
            state_names=combined_state.state_names,
            name=name,
        )
        model_state.cross_module_transitions = extract_cross_module_transitions(model_state, module_offsets)
        return model_state, PomegranateModernHistory()

    def _create_combined_state(
        self,
        *,
        trained_models: dict[str, BaseTrainableHmm],
        labels_train_sequence: list[np.ndarray],
        model_config: CompositeHmmConfig,
        module_offsets: dict[str, int],
        initialization: Literal["labels", "fully-connected"],
        name: str,
    ) -> FlatHmmState:
        n_states = sum(module.n_states for module in model_config.modules)
        if initialization == "fully-connected":
            transition_probs, start_probs, end_probs = create_transition_matrix_fully_connected(n_states)
            return FlatHmmState(
                graph=HmmGraphState(transition_probs=transition_probs, start_probs=start_probs, end_probs=end_probs),
                emissions=tuple(
                    freeze_emission(emission)
                    for module in model_config.modules
                    for emission in trained_models[module.name].model.emissions
                ),
                state_names=create_state_names(n_states),
                name=name,
            )

        transition_probs = np.zeros((n_states, n_states), dtype=float)
        emissions = []
        for module in model_config.modules:
            module_state = trained_models[module.name].model
            offset = module_offsets[module.name]
            size = module.n_states
            transition_probs[offset : offset + size, offset : offset + size] = module_state.graph.transition_probs
            emissions.extend(freeze_emission(emission) for emission in module_state.emissions)

        transitions, _, _ = extract_transitions_starts_stops_from_hidden_state_sequence(labels_train_sequence)
        start_probs, end_probs = estimate_sequence_boundary_probs(labels_train_sequence, n_states)
        for from_state, to_state in transitions:
            transition_probs[int(from_state[1:]), int(to_state[1:])] = max(
                transition_probs[int(from_state[1:]), int(to_state[1:])],
                0.1,
            )

        transition_probs, end_probs = normalize_transition_and_end_probs(transition_probs, end_probs)

        return FlatHmmState(
            graph=HmmGraphState(transition_probs=transition_probs, start_probs=start_probs, end_probs=end_probs),
            emissions=tuple(emissions),
            state_names=create_state_names(n_states),
            name=name,
        )
