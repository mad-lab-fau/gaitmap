"""Shared helpers for multiple HMM backends."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gaitmap_mad.stride_segmentation.hmm._state import CrossModuleTransition, HMMState
from gaitmap_mad.stride_segmentation.hmm._utils import _DataToShortError


def extract_cross_module_transitions(
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


def prepare_predict_data(data: pd.DataFrame, expected_columns: tuple[str, ...], n_states: int) -> np.ndarray:
    try:
        data = data[list(expected_columns)]
    except KeyError as e:
        raise ValueError(
            "The provided feature data is expected to have the following columns:\n\n"
            f"{expected_columns}\n\n"
            "But it only has the following columns:\n\n"
            f"{data.columns}"
        ) from e

    if len(data) < n_states:
        raise _DataToShortError(
            "The provided feature data is expected to have at least as many samples as the number of states "
            f"of the model ({n_states}). "
            f"But it only has {len(data)} samples."
        )
    return np.ascontiguousarray(data.to_numpy())


def normalize_transition_and_end_probs(
    transition_probs: np.ndarray, end_probs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    normalized_transitions = np.asarray(transition_probs, dtype=float).copy()
    normalized_ends = np.asarray(end_probs, dtype=float).copy()
    for row_idx in range(len(normalized_transitions)):
        total = normalized_transitions[row_idx].sum() + normalized_ends[row_idx]
        if total <= 0:
            continue
        normalized_transitions[row_idx] /= total
        normalized_ends[row_idx] /= total
    return normalized_transitions, normalized_ends
