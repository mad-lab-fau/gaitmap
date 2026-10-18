"""Utils and helper functions for HMM classes."""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
from tpcp import BaseTpcpObject, CloneFactory

from gaitmap.utils.datatype_helper import SingleSensorData, SingleSensorRegionsOfInterestList, SingleSensorStrideList
from gaitmap_mad.stride_segmentation.hmm._hmm_distributions import fit_distribution
from gaitmap_mad.stride_segmentation.hmm._hmm_inference import HiddenMarkovModel
from gaitmap_mad.stride_segmentation.hmm._hmm_training import History


class ShortenedHMMPrint(BaseTpcpObject):
    """Mixin class to better format HMM models when printing them."""

    def __repr_parameter__(self, name: str, value: Any) -> str:
        """Representation with specific care for HMM models."""
        if name == "model":
            if isinstance(value, HiddenMarkovModel):
                return f"{name}=HiddenMarkovModel[name={value.name}](...)"
            if isinstance(value, CloneFactory) and (isinstance(value.default_value, HiddenMarkovModel)):
                return f"{name}=cf(HiddenMarkovModel[name={value.get_value().name}](...))"
        return super().__repr_parameter__(name, value)


def create_transition_matrix_fully_connected(n_states: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create nxn transition matrix with only 1 entries."""
    transition_matrix = np.ones((n_states, n_states)) / n_states
    start_probs = np.ones(n_states)
    end_probs = np.ones(n_states)

    return transition_matrix, start_probs, end_probs


def create_transition_matrix_left_right(
    n_states: int, self_transition: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create nxn transition for left to right model."""
    transition_matrix = np.zeros((n_states, n_states))
    transition_matrix[range(n_states - 1), range(1, n_states)] = 1
    transition_matrix[range(n_states), range(n_states)] = 1
    if self_transition:
        transition_matrix[-1][0] = 1

    # force start with first state
    start_probs = np.zeros(n_states)
    start_probs[0] = 1
    # and force end with last state
    end_probs = np.zeros(n_states)
    end_probs[-1] = 1

    return transition_matrix, start_probs, end_probs


def print_transition_matrix(model: HiddenMarkovModel, precision: int = 3) -> None:
    """Print model transition matrix in user-friendly format."""
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision)
    if isinstance(model, np.ndarray):
        print(model)
    else:
        print(model.dense_transition_matrix()[0:-2, 0:-2])


def cluster_data_by_labels(data_list: list[np.ndarray], label_list: list[np.ndarray]):
    """Cluster data by labels."""
    assert isinstance(label_list, list), "label_list must be list!"
    assert isinstance(data_list, list), "data_list must be list!"

    label_list = [np.asarray(label).tolist() for label in label_list]
    data_list = [np.asarray(data).tolist() for data in data_list]

    # remove datasets where the labellist is None
    x_ = [x for x, label in zip(data_list, label_list) if label is not None]
    x_ = np.concatenate(x_)  # concatenate all datasets with label_list to a single array

    labels_ = np.concatenate(
        [label for label in label_list if label is not None]
    )  # concatenate all not None labellists to a single array
    label_set = np.unique(labels_)  # get set of unique label_list

    clustered_data = [x_[labels_ == label] for label in label_set]

    return clustered_data


def gmms_from_samples(
    data,
    labels,
    n_components: int,
    n_expected_states: int,
    verbose: bool = False,
    n_init: int = 5,
    n_jobs: int = 1,
    min_covar: float = 1e-6,
):
    """Create Gaussian Mixture Models from samples.

    This function clusters the data by the given labels and fits either univariate or multivariate
    Normal Distributions for each cluster. If n_components is > 1 then a Mixture Model of n-univariate or n-multivariate
    Gaussian Distributions will be fitted.
    """
    if not np.array(data, dtype=object).data.c_contiguous:
        raise ValueError("Memory Layout of given input data is not contiguous! Consider using numpy.ascontiguousarray.")

    clustered_data = cluster_data_by_labels(data, labels)

    if len(clustered_data) < n_expected_states:
        raise ValueError(
            f"The training labels did only provide samples for {len(clustered_data)} states, but "
            f"{n_expected_states} states were expected. "
            "Ensure that the training data contains samples for all states."
        )

    for cluster in clustered_data:
        if len(cluster) < n_components:
            raise ValueError(
                f"The training labels did only provide a small number of samples ({len(cluster)}) for one of the "
                "states. "
                f"To initialize {n_components} components in a mixture model, we need at least {n_components} "
                "samples! Ensure that the training data contains enough samples."
            )
    distributions = [
        fit_distribution(
            cluster, n_components=n_components, verbose=verbose, n_init=n_init, n_jobs=n_jobs, min_covar=min_covar
        )
        for cluster in clustered_data
    ]
    return distributions, clustered_data


def _iter_nested_distributions(distribution):
    if hasattr(distribution, "distributions"):
        yield distribution
        for d in distribution.distributions:
            yield from _iter_nested_distributions(d)
    else:
        yield distribution


def model_params_are_finite(model: HiddenMarkovModel) -> bool:
    """Check if model parameters are finite."""
    for state in model.states:
        for dist in _iter_nested_distributions(state.distribution):
            if hasattr(dist, "parameters"):
                for param in dist.parameters:
                    if not np.all(np.isfinite(param)):
                        return False
            if hasattr(dist, "weights") and not np.all(np.isfinite(dist.weights)):
                return False
    return True


def check_history_for_training_failure(history: History) -> None:
    """Check if training history contains any NaNs."""
    if not np.all(np.isfinite(history.improvements)) or np.any(np.array(history.improvements) < 0):
        warnings.warn(
            "During training the improvement per epoch became NaN/infinite or negative! "
            "Run `self_optimize_with_info` and inspect the history element for more information. "
            "With a high likelihood, the final model is not usable and will result in errors during prediction. "
            "This usually happens when there is not enough data for a large number of distributions and "
            "states. "
            "To avoid this issue, reduce the number of distributions per state or the number of states. "
            "Or ideally, provide more data."
        )


def get_model_distributions(model: HiddenMarkovModel) -> list:
    """Return all not None distributions as list from given model."""
    distributions = []
    for state in model.states:
        if state.distribution is not None:
            distributions.append(state.distribution)
    return distributions


def extract_transitions_starts_stops_from_hidden_state_sequence(
    hidden_state_sequence: list[np.ndarray],
) -> tuple[set[tuple[int, int]], np.ndarray, np.ndarray]:
    """Extract every observed transition, including self-loops, and start/stop labels."""
    assert isinstance(hidden_state_sequence, list), "Hidden state sequence must be list!"

    transitions = set()
    starts = []
    ends = []
    for labels in hidden_state_sequence:
        starts.append(labels[0])
        ends.append(labels[-1])
        for source, target in zip(labels[:-1], labels[1:]):
            transitions.add((int(source), int(target)))

    starts = np.unique(starts).astype("int64")
    ends = np.unique(ends).astype("int64")

    return transitions, starts, ends


def create_equidistant_label_sequence(n_labels: int, n_states: int) -> np.ndarray:
    """Create equidistant label sequence.

    create label sequence of length n_states with n_labels unique labels.
    This can be used to e.g. initialize labels for a single stride or sequence that is expected to be left-right strict.

    In case n_labels is not cleanly dividable by n_states, some of the states will be repeated to ensure that the
    sequence is of length n_labels.
    Specifically, we will repeat states at the start and the end of the sequence.

    If the number of labels is smaller than the number of states, an error is raised.

    Parameters
    ----------
    n_labels : int
        Number of labels to create.
    n_states : int
        Number of unique states in the output sequence.

    Example
    -------
    >>> create_equidistant_label_sequence(n_labels=10, n_states=5)
    array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    >>> create_equidistant_label_sequence(n_labels=10, n_states=4)
    array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3])
    >>> create_equidistant_label_sequence(n_labels=10, n_states=3)
    array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])

    """
    if n_labels < n_states:
        raise ValueError("n_labels must be larger than n_states!")

    # calculate the samples per state (must be integer!)
    save_repeats = int(n_labels // n_states)
    remainder = n_labels % n_states

    # create label sequence
    max_handled_state = remainder // 2
    label_sequence = np.repeat(np.arange(remainder // 2), save_repeats + 1)
    label_sequence = np.append(
        label_sequence, np.repeat(np.arange(n_states - remainder) + max_handled_state, save_repeats)
    )
    max_handled_state = n_states - remainder + max_handled_state
    label_sequence = np.append(
        label_sequence, np.repeat(np.arange(n_states - max_handled_state) + max_handled_state, save_repeats + 1)
    )

    return label_sequence


def convert_stride_list_to_transition_list(
    stride_list: SingleSensorStrideList, last_end: int
) -> SingleSensorRegionsOfInterestList:
    """Extract the regions between strides as transitions from a stride list.

    Parameters
    ----------
    stride_list
        Stride list to extract transitions from.
    last_end
        End of the final transition (usually len(data)).

    """
    # Transitions are everything between two strides
    transition_starts = [0, *(stride_list["end"] + 1)]
    transition_ends = [*stride_list["start"], last_end]

    return pd.DataFrame(
        [(s, e) for s, e in zip(transition_starts, transition_ends) if e - s > 0], columns=["start", "end"]
    )


def get_train_data_sequences_transitions(
    data_train_sequence: list[SingleSensorData], stride_list_sequence: list[SingleSensorStrideList], n_states: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract Transition Training set.

    - data_train_sequence: list of datasets in feature space
    - stride_list_sequence: list of gaitmap stride-lists
    - n_states: number of labels.
    """
    trans_data_train_sequence = []
    trans_labels_train_sequence = []

    n_too_short_transitions = 0

    for data, stride_list in zip(data_train_sequence, stride_list_sequence):
        # for each transition, get data and create some naive labels for initialization
        for start, end in convert_stride_list_to_transition_list(stride_list, data.shape[0])[
            ["start", "end"]
        ].to_numpy():
            # append extracted sequences and corresponding label set to results list
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
            "than the expected number of transition states ({n_states}). "
            "This warning can usually be ignored, if the number of remaining transitions is still large "
            "enough to train a model."
        )

    return trans_data_train_sequence, trans_labels_train_sequence


def get_train_data_sequences_strides(
    data_train_sequence: list[SingleSensorData], stride_list_sequence: list[SingleSensorStrideList], n_states: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract Transition Training set.

    - data_train_sequence: list of datasets in feature space
    - stride_list_sequence: list of gaitmap stride-lists
    - n_states: number of labels.
    """
    stride_data_train_sequence = []
    stride_labels_train_sequence = []

    n_too_short_strides = 0

    for data, stride_list in zip(data_train_sequence, stride_list_sequence):
        # extract strides directly from stride_list
        for start, end in stride_list[["start", "end"]].to_numpy():
            try:
                labels = create_equidistant_label_sequence(end - start, n_states).astype("int64")
            except ValueError:
                n_too_short_strides += 1
                continue
            stride_labels_train_sequence.append(labels)
            stride_data_train_sequence.append(data[start:end])

    if n_too_short_strides > 0:
        warnings.warn(
            f"{n_too_short_strides} strides (out of {len(stride_data_train_sequence) + n_too_short_strides}) "
            f"were ignored, because they were shorter than the expected number of stride states ({n_states}). "
            "This warning can usually be ignored, if the number of remaining strides is still large "
            "enough to train a model."
        )

    return stride_data_train_sequence, stride_labels_train_sequence


class _DataToShortError(ValueError):
    pass


def predict(
    model: HiddenMarkovModel | None,
    data: pd.DataFrame,
    *,
    expected_columns: tuple[str, ...],
    algorithm: Literal["viterbi", "map"],
) -> np.ndarray:
    """Predict the hidden state sequence for the given data.

    Parameters
    ----------
    model
        The hidden markov model to use for prediction.
    data
        The data to predict the hidden state sequence for.
    expected_columns
        The expected columns of the data.
        This is used to check if the data has the correct format and re-order the columns if necessary.
    algorithm
        The algorithm to use for prediction.

    Returns
    -------
    hidden_state_sequence
        A numpy array containing the predicted hidden state sequence.

    """
    if model is None:
        raise ValueError(
            "You need to train the HMM before calling `predict_hidden_state_sequence`. "
            "Use `self_optimize` or `self_optimize_with_info` for that."
        )

    try:
        data = data[list(expected_columns)]
    except KeyError as e:
        raise ValueError(
            "The provided feature data is expected to have the following columns:\n\n"
            f"{expected_columns}\n\n"
            "But it only has the following columns:\n\n"
            f"{data.columns}"
        ) from e

    if len(data) < len(model.states) - 2:
        raise _DataToShortError(
            "The provided feature data is expected to have at least as many samples as the number of states "
            f"of the model ({len(model.states) - 2}). "
            f"But it only has {len(data)} samples."
        )

    data = np.ascontiguousarray(data.to_numpy())
    try:
        labels_predicted = np.asarray(model.predict(data.copy(), algorithm=algorithm))
    except Exception as e:
        if not model_params_are_finite(model):
            raise ValueError(
                "Prediction failed! (See error above.). "
                "However, the provided HMM model has non-finite/NaN parameters. "
                "This might be the source of the observed error and indicates problems during training. "
                "Check the training history and the model parameters to confirm invalid training behaviour. "
                "Unfortunately, there is no way to automatically fix these issues. "
                "Simply speaking, your training data could not be represented well by the selected model architecture. "
                "Check for obvious errors in your pre-processing or try to use a different model architecture. "
            ) from e
        raise ValueError(
            "Prediction failed! (See error above.). "
            "Unfortunately, we are not sure what happened. "
            "The error was caused by the HMM prediction backend."
        ) from e

    # Preserve gaitmap's historical Viterbi output length. Legacy finite paths include start/end sentinels;
    # paths without an explicit end include only the start sentinel, so gaitmap also dropped the final emission.
    # Keeping that length matters when the feature labels are resampled back to the input sampling rate.
    if algorithm == "viterbi" and not model.finite:
        labels_predicted = labels_predicted[:-1]
    return np.asarray(labels_predicted)
