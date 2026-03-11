"""Legacy pomegranate backend."""

from __future__ import annotations

import copy
from importlib import import_module
from typing import Any, Literal

import numpy as np
import pandas as pd

from gaitmap_mad.stride_segmentation.hmm._backend_base import BaseHmmBackend
from gaitmap_mad.stride_segmentation.hmm._backend_common import (
    extract_cross_module_transitions,
    normalize_transition_and_end_probs,
)
from gaitmap_mad.stride_segmentation.hmm._config import CompositeHmmConfig, HmmSubModelConfig
from gaitmap_mad.stride_segmentation.hmm._pomegranate import create_state_names
from gaitmap_mad.stride_segmentation.hmm._state import (
    BackendInfo,
    HMMState,
    HmmSubModelState,
    hmm_state_to_pomegranate_model,
    pomegranate_model_to_flat_hmm_state,
    pomegranate_model_to_hmm_state,
)
from gaitmap_mad.stride_segmentation.hmm._utils import (
    _clone_model,
    check_history_for_training_failure,
    create_transition_matrix_fully_connected,
    estimate_sequence_boundary_probs,
    extract_transitions_starts_stops_from_hidden_state_sequence,
    fix_model_names,
    get_model_distributions,
    labels_to_strings,
    predict,
)


def _create_simple_hmm_from_config(config: HmmSubModelConfig) -> Any:
    simple_hmm = import_module("gaitmap_mad.stride_segmentation.hmm._simple_model").SimpleHmm
    return simple_hmm(
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
        super().__init__(backend_id=backend_id)

    def create_submodel(self, config: HmmSubModelConfig) -> Any:
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
        runtime_model = hmm_state_to_pomegranate_model(model, verbose=verbose)
        return predict(runtime_model, data, expected_columns=expected_columns, algorithm=algorithm)

    def finalize_model(
        self,
        *,
        trained_models: dict[str, Any],
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
        trained_models: dict[str, Any],
        labels_train_sequence: list[np.ndarray],
        distributions: list[Any],
        model_config: CompositeHmmConfig,
        module_offsets: dict[str, int],
        initialization: Literal["labels", "fully-connected"],
        verbose: bool,
    ) -> Any:
        pg = import_module("pomegranate")

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
                trans_mat[
                    offset : offset + module.n_states,
                    offset : offset + module.n_states,
                ] = module_transition_matrix

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
