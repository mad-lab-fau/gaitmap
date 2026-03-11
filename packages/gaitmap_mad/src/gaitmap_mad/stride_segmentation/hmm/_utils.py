"""Backend-neutral helper utilities for HMM preprocessing and label handling."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from gaitmap.utils.datatype_helper import (
    SingleSensorData,
    SingleSensorRegionsOfInterestList,
    SingleSensorStrideList,
    is_single_sensor_regions_of_interest_list,
)
from gaitmap.utils.exceptions import ValidationError


def create_transition_matrix_fully_connected(n_states: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a fully-connected transition matrix with uniform probabilities."""
    transition_matrix = np.ones((n_states, n_states)) / n_states
    start_probs = np.ones(n_states)
    end_probs = np.ones(n_states)
    return transition_matrix, start_probs, end_probs


def create_transition_matrix_left_right(
    n_states: int, self_transition: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a left-right transition matrix."""
    transition_matrix = np.zeros((n_states, n_states))
    transition_matrix[range(n_states - 1), range(1, n_states)] = 1
    transition_matrix[range(n_states), range(n_states)] = 1
    if self_transition:
        transition_matrix[-1][0] = 1

    start_probs = np.zeros(n_states)
    start_probs[0] = 1
    end_probs = np.zeros(n_states)
    end_probs[-1] = 1

    return transition_matrix, start_probs, end_probs


def create_state_names(n_states: int) -> tuple[str, ...]:
    """Create stable state names matching the legacy naming scheme."""
    state_names = []
    for state_number in range(n_states):
        if state_number < 10:
            state_names.append(f"s{state_number}")
            continue
        state_names.append(f"s{chr(87 + state_number)}")
    return tuple(state_names)


def cluster_data_by_labels(data_list: list[np.ndarray], label_list: list[np.ndarray]):
    """Cluster data arrays by integer labels."""
    assert isinstance(label_list, list), "label_list must be list!"
    assert isinstance(data_list, list), "data_list must be list!"

    label_list = [np.asarray(label).tolist() for label in label_list]
    data_list = [np.asarray(data).tolist() for data in data_list]

    x_ = [x for x, label in zip(data_list, label_list) if label is not None]
    x_ = np.concatenate(x_)
    labels_ = np.concatenate([label for label in label_list if label is not None])
    label_set = np.unique(labels_)
    return [x_[labels_ == label] for label in label_set]


def extract_transitions_starts_stops_from_hidden_state_sequence(
    hidden_state_sequence: list[np.ndarray],
) -> tuple[set[tuple[str, str]], np.ndarray, np.ndarray]:
    """Extract observed transitions and start/end states from labeled sequences."""
    assert isinstance(hidden_state_sequence, list), "Hidden state sequence must be list!"

    transitions = []
    starts = []
    ends = []
    for labels in hidden_state_sequence:
        starts.append(labels[0])
        ends.append(labels[-1])
        for idx in np.where(abs(np.diff(labels)) > 0)[0]:
            transitions.append((f"s{int(labels[idx])}", f"s{int(labels[idx + 1])}"))

    if len(transitions) > 0:
        transitions = set(transitions)
    starts = np.unique(starts).astype("int64")
    ends = np.unique(ends).astype("int64")

    return transitions, starts, ends


def estimate_sequence_boundary_probs(
    hidden_state_sequence: list[np.ndarray], n_states: int
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate start and end probabilities from sequence boundary counts."""
    if n_states <= 0:
        raise ValueError("`n_states` must be positive.")

    start_counts = np.zeros(n_states, dtype=float)
    end_counts = np.zeros(n_states, dtype=float)
    for labels in hidden_state_sequence:
        start_counts[int(labels[0])] += 1.0
        end_counts[int(labels[-1])] += 1.0

    start_total = start_counts.sum()
    end_total = end_counts.sum()
    if start_total <= 0 or end_total <= 0:
        raise ValueError("At least one non-empty hidden-state sequence is required to estimate boundary probabilities.")

    return start_counts / start_total, end_counts / end_total


def create_equidistant_label_sequence(n_labels: int, n_states: int) -> np.ndarray:
    """Create a length-`n_labels` sequence with `n_states` approximately equidistant labels."""
    if n_labels < n_states:
        raise ValueError("n_labels must be larger than n_states!")

    safe_repeats = int(n_labels // n_states)
    remainder = n_labels % n_states

    max_handled_state = remainder // 2
    label_sequence = np.repeat(np.arange(remainder // 2), safe_repeats + 1)
    label_sequence = np.append(
        label_sequence, np.repeat(np.arange(n_states - remainder) + max_handled_state, safe_repeats)
    )
    max_handled_state = n_states - remainder + max_handled_state
    label_sequence = np.append(
        label_sequence, np.repeat(np.arange(n_states - max_handled_state) + max_handled_state, safe_repeats + 1)
    )

    return label_sequence


def convert_stride_list_to_transition_list(
    stride_list: SingleSensorStrideList, last_end: int
) -> SingleSensorRegionsOfInterestList:
    """Extract the regions between strides as transitions from a stride list."""
    transition_starts = [0, *(stride_list["end"] + 1)]
    transition_ends = [*stride_list["start"], last_end]

    return pd.DataFrame(
        [(start, end) for start, end in zip(transition_starts, transition_ends) if end - start > 0],
        columns=["start", "end"],
    )


def validate_trainable_region_list(
    region_list: SingleSensorRegionsOfInterestList, explicit_region_types: tuple[str, ...]
) -> SingleSensorRegionsOfInterestList:
    """Validate and normalize a typed region list for HMM training."""
    try:
        is_single_sensor_regions_of_interest_list(region_list, region_type="any", raise_exception=True)
        normalized_region_list = region_list.reset_index()
        if "type" not in normalized_region_list.columns:
            raise ValidationError("The region list is expected to have a `type` column.")
        if normalized_region_list["type"].isna().any():
            raise ValidationError("The `type` column of the region list is not allowed to contain missing values.")
        invalid_types = sorted(set(normalized_region_list["type"]) - set(explicit_region_types))
        if invalid_types:
            raise ValidationError(
                f"The region list contains unknown region types {invalid_types}. "
                f"Expected only the following explicit region types: {list(explicit_region_types)}"
            )
        normalized_region_list = normalized_region_list.sort_values(["start", "end"]).reset_index(drop=True)
        regions = normalized_region_list[["start", "end"]].to_numpy()
        if np.any(regions[:, 1] <= regions[:, 0]):
            raise ValidationError("All regions must satisfy `end > start`.")
        if len(regions) > 1 and np.any(regions[1:, 0] < regions[:-1, 1]):
            raise ValidationError("The region list must not contain overlapping regions.")
    except ValidationError as e:
        raise ValidationError(
            "The passed object does not seem to be a valid typed region list for HMM training. "
            f"The validation failed with the following error:\n\n{e!s}"
        ) from e
    return normalized_region_list


def convert_region_list_to_transition_list(
    region_list: SingleSensorRegionsOfInterestList, last_end: int
) -> SingleSensorRegionsOfInterestList:
    """Return the implicit transition regions, i.e. everything not covered by explicit regions."""
    regions = region_list[["start", "end"]].sort_values(["start", "end"]).to_numpy()
    transition_regions = []
    current_start = 0
    for start, end in regions:
        if start > current_start:
            transition_regions.append((current_start, start))
        current_start = end
    if current_start < last_end:
        transition_regions.append((current_start, last_end))
    return pd.DataFrame(transition_regions, columns=["start", "end"])


def get_train_data_sequences_transitions(
    data_train_sequence: list[SingleSensorData],
    region_list_sequence: list[SingleSensorRegionsOfInterestList],
    n_states: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract transition training sequences and naive initial labels."""
    trans_data_train_sequence = []
    trans_labels_train_sequence = []
    n_too_short_transitions = 0

    for data, region_list in zip(data_train_sequence, region_list_sequence):
        transition_regions = convert_region_list_to_transition_list(region_list, data.shape[0])
        for start, end in transition_regions[["start", "end"]].to_numpy():
            try:
                labels = create_equidistant_label_sequence(end - start, n_states).astype("int64")
            except ValueError:
                n_too_short_transitions += 1
                continue
            trans_labels_train_sequence.append(labels)
            trans_data_train_sequence.append(data[start:end])

    if n_too_short_transitions > 0:
        warnings.warn(
            f"{n_too_short_transitions} transitions (out of "
            f"{len(trans_labels_train_sequence) + n_too_short_transitions}) were ignored, because they were shorter "
            f"than the expected number of transition states ({n_states}). "
            "This warning can usually be ignored, if the number of remaining transitions is still large "
            "enough to train a model."
        )

    return trans_data_train_sequence, trans_labels_train_sequence


def get_train_data_sequences_regions(
    data_train_sequence: list[SingleSensorData],
    region_list_sequence: list[SingleSensorRegionsOfInterestList],
    region_type: str,
    n_states: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract training sequences for one explicit region type."""
    region_data_train_sequence = []
    region_labels_train_sequence = []
    n_too_short_regions = 0

    for data, region_list in zip(data_train_sequence, region_list_sequence):
        matching_regions = region_list[region_list["type"] == region_type]
        for start, end in matching_regions[["start", "end"]].to_numpy():
            try:
                labels = create_equidistant_label_sequence(end - start, n_states).astype("int64")
            except ValueError:
                n_too_short_regions += 1
                continue
            region_labels_train_sequence.append(labels)
            region_data_train_sequence.append(data[start:end])

    if n_too_short_regions > 0:
        warnings.warn(
            f"{n_too_short_regions} regions of type `{region_type}` "
            f"(out of {len(region_data_train_sequence) + n_too_short_regions}) were ignored, because they were "
            f"shorter than the expected number of hidden states ({n_states}). "
            "This warning can usually be ignored, if the number of remaining regions is still large "
            "enough to train a model."
        )

    return region_data_train_sequence, region_labels_train_sequence


class _DataToShortError(ValueError):
    """Raised when feature-space input is shorter than the model topology requires."""
