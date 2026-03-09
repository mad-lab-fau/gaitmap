"""Backend abstractions for HMM training and inference."""

from __future__ import annotations

import copy
from typing import Literal

import numpy as np
import pandas as pd
import pomegranate as pg
from pomegranate.hmm import History

from gaitmap.base import _BaseSerializable
from gaitmap_mad.stride_segmentation.hmm._config import CompositeHmmConfig, HmmSubModelConfig
from gaitmap_mad.stride_segmentation.hmm._simple_model import SimpleHmm
from gaitmap_mad.stride_segmentation.hmm._state import (
    BackendInfo,
    CrossModuleTransition,
    HMMState,
    HmmSubModelState,
    hmm_state_to_pomegranate_model,
    pomegranate_model_to_flat_hmm_state,
    pomegranate_model_to_hmm_state,
)
from gaitmap_mad.stride_segmentation.hmm._utils import (
    _clone_model,
    add_transition,
    check_history_for_training_failure,
    create_transition_matrix_fully_connected,
    extract_transitions_starts_stops_from_hidden_state_sequence,
    fix_model_names,
    get_model_distributions,
    labels_to_strings,
    predict,
)


class BaseHmmBackend(_BaseSerializable):
    """Base abstraction for backend-specific HMM primitives."""

    backend_id: str

    def __init__(self, backend_id: str) -> None:
        self.backend_id = backend_id

    def create_submodel(self, config: HmmSubModelConfig) -> SimpleHmm:
        """Create a backend-specific trainable flat HMM wrapper."""
        raise NotImplementedError

    def predict(
        self,
        model: HMMState,
        data: pd.DataFrame,
        *,
        expected_columns: tuple[str, ...],
        algorithm: Literal["viterbi", "map"],
        verbose: bool,
    ) -> np.ndarray:
        """Predict hidden states with a serialized model."""
        raise NotImplementedError

    def finalize_model(
        self,
        *,
        trained_models: dict[str, SimpleHmm],
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
    ) -> tuple[HMMState, History]:
        """Create, train, and serialize the final combined HMM."""
        raise NotImplementedError


def _create_simple_hmm_from_config(config: HmmSubModelConfig) -> SimpleHmm:
    return SimpleHmm(
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


def _build_submodel_states(
    model_config: CompositeHmmConfig, trained_models: dict[str, SimpleHmm]
) -> tuple[HmmSubModelState, ...]:
    return tuple(
        HmmSubModelState(
            name=module.name,
            role=module.role,
            model=pomegranate_model_to_flat_hmm_state(trained_models[module.name].model),
        )
        for module in model_config.modules
    )


def _extract_cross_module_transitions(
    compiled_state: HMMState,
    module_offsets: dict[str, int],
) -> tuple[CrossModuleTransition, ...]:
    transitions = []
    transition_matrix = compiled_state.compiled.graph.transition_probs
    module_sizes = {submodel.name: len(submodel.model.state_names) for submodel in compiled_state.submodels}
    ordered_modules = tuple(submodel.name for submodel in compiled_state.submodels)
    for from_module in ordered_modules:
        from_offset = module_offsets[from_module]
        from_size = module_sizes[from_module]
        for to_module in ordered_modules:
            if from_module == to_module:
                continue
            to_offset = module_offsets[to_module]
            to_size = module_sizes[to_module]
            for from_state in range(from_size):
                for to_state in range(to_size):
                    probability = transition_matrix[from_offset + from_state, to_offset + to_state]
                    if probability <= 0:
                        continue
                    transitions.append(
                        CrossModuleTransition(
                            from_module=from_module,
                            from_state=from_state,
                            to_module=to_module,
                            to_state=to_state,
                            probability=float(probability),
                        )
                    )
    return tuple(transitions)


class PomegranateHmmBackend(BaseHmmBackend):
    """`pomegranate 0.14` backend for HMM training and inference."""

    def __init__(self, backend_id: str = "pomegranate-legacy") -> None:
        super().__init__(backend_id=backend_id)

    def create_submodel(self, config: HmmSubModelConfig) -> SimpleHmm:
        """Create a pomegranate-backed trainable submodel."""
        return _create_simple_hmm_from_config(config)

    def predict(
        self,
        model: HMMState,
        data: pd.DataFrame,
        *,
        expected_columns: tuple[str, ...],
        algorithm: Literal["viterbi", "map"],
        verbose: bool,
    ) -> np.ndarray:
        """Compile the serialized state and predict hidden states."""
        runtime_model = hmm_state_to_pomegranate_model(model, verbose=verbose)
        return predict(runtime_model, data, expected_columns=expected_columns, algorithm=algorithm)

    def finalize_model(
        self,
        *,
        trained_models: dict[str, SimpleHmm],
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
    ) -> tuple[HMMState, History]:
        """Build the final pomegranate model, train it, and convert it to `HMMState`."""
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

        submodel_states = _build_submodel_states(model_config, trained_models)
        model_state = pomegranate_model_to_hmm_state(
            model,
            submodels=submodel_states,
            backend_info=BackendInfo(backend_id=self.backend_id),
        )
        model_state.cross_module_transitions = _extract_cross_module_transitions(model_state, module_offsets)
        return model_state, history

    def _create_combined_model(
        self,
        *,
        trained_models: dict[str, SimpleHmm],
        labels_train_sequence: list[np.ndarray],
        distributions: list[pg.Distribution],
        model_config: CompositeHmmConfig,
        module_offsets: dict[str, int],
        initialization: Literal["labels", "fully-connected"],
        verbose: bool,
    ) -> pg.HiddenMarkovModel:
        n_states = sum(module.n_states for module in model_config.modules)
        if initialization == "fully-connected":
            trans_mat, start_probs, _end_probs = create_transition_matrix_fully_connected(n_states)
            model = pg.HiddenMarkovModel.from_matrix(
                transition_probabilities=copy.deepcopy(trans_mat),
                distributions=copy.deepcopy(distributions),
                starts=start_probs,
                ends=None,
                state_names=None,
                verbose=verbose,
            )
        else:
            trans_mat = np.zeros((n_states, n_states))
            for module in model_config.modules:
                module_transition_matrix = trained_models[module.name].model.dense_transition_matrix()[:-2, :-2]
                offset = module_offsets[module.name]
                trans_mat[
                    offset : offset + module.n_states,
                    offset : offset + module.n_states,
                ] = module_transition_matrix

            transitions, starts, _ = extract_transitions_starts_stops_from_hidden_state_sequence(labels_train_sequence)

            start_probs = np.zeros(n_states)
            start_probs[starts] = 1.0

            model = pg.HiddenMarkovModel.from_matrix(
                transition_probabilities=copy.deepcopy(trans_mat),
                distributions=copy.deepcopy(distributions),
                starts=start_probs,
                ends=None,
                state_names=None,
                verbose=verbose,
            )

            existing_transitions = {(start.name, end.name) for start, end in model.graph.edges()}
            for transition in sorted(transitions - existing_transitions):
                add_transition(model, transition, 0.1)

        model = fix_model_names(model)
        model.bake()
        model.freeze_distributions()
        return _clone_model(model, assert_correct=False)
