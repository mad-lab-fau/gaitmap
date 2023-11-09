"""Utils and helper functions for HMM classes."""
import warnings
from typing import Any, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from pomegranate._utils import _update_parameter
from pomegranate.distributions import Normal
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.hmm import DenseHMM as pgHMM
from tpcp import BaseTpcpObject, CloneFactory

from gaitmap.utils.datatype_helper import (
    SingleSensorData,
    SingleSensorRegionsOfInterestList,
    SingleSensorStrideList,
)


class RobustNormal(Normal):
    def from_summaries(self):
        """Update the model parameters given the extracted statistics.

        This method uses calculated statistics from calls to the `summarize`
        method to update the distribution parameters. Hyperparameters for the
        update are passed in at initialization time.

        Note: Internally, a call to `fit` is just a successive call to the
        `summarize` method followed by the `from_summaries` method.
        """
        if self.frozen is True:
            return

        means = self._xw_sum / self._w_sum

        if self.covariance_type == "full":
            v = self._xw_sum.unsqueeze(0) * self._xw_sum.unsqueeze(1)
            covs = self._xxw_sum / self._w_sum - v / self._w_sum**2.0

            covs += np.eye(covs.shape[0]) * 1e-6

        elif self.covariance_type in ["diag", "sphere"]:
            covs = self._xxw_sum / self._w_sum - self._xw_sum**2.0 / self._w_sum**2.0
            if self.covariance_type == "sphere":
                covs = covs.mean(dim=-1)

            covs += 1e-6

        _update_parameter(self.means, means, self.inertia)
        _update_parameter(self.covs, covs, self.inertia)
        self._reset_cache()


class ShortenedHMMPrint(BaseTpcpObject):
    """Mixin class to better format pg.HMM models when printing them."""

    def __repr_parameter__(self, name: str, value: Any) -> str:
        """Representation with specific care for HMM models."""
        if name == "model":
            if isinstance(value, pgHMM):
                return f"{name}=HiddenMarkovModel(...)"
            if isinstance(value, CloneFactory) and isinstance(value.default_value, pgHMM):
                return f"{name}=cf(HiddenMarkovModel(...))"
        return super().__repr_parameter__(name, value)


def create_transition_matrix_fully_connected(
    n_states: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create nxn transition matrix with only 1 entries."""
    transition_matrix = np.ones((n_states, n_states))
    start_probs = np.ones(n_states)
    end_probs = np.ones(n_states)

    return transition_matrix, start_probs, end_probs


def create_transition_matrix_left_right(
    n_states: int, self_transition: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def create_transition_matrix_left_right_loose(
    n_states: int, self_transition: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create nxn transition for left to right model."""
    transition_matrix = np.zeros((n_states, n_states))
    transition_matrix[range(n_states - 1), range(1, n_states)] = 1
    transition_matrix[range(n_states), range(n_states)] = 1
    if self_transition:
        transition_matrix[-1][0] = 1

    start_probs = np.ones(n_states).astype(np.float32)
    end_probs = np.ones(n_states).astype(np.float32)

    return transition_matrix, start_probs, end_probs


def cluster_data_by_labels(data_list: List[np.ndarray], label_list: List[np.ndarray]):
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

    clustered_data = [torch.as_tensor(x_[labels_ == label], dtype=torch.float32) for label in label_set]

    return clustered_data


def gmms_from_samples(
    data,
    labels,
    n_components: int,
    n_expected_states: int,
    verbose: bool = False,
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

    if data[0].ndim == 1:
        # we need to reshape the 1D-data for pomegranate!
        clustered_data = [np.reshape(data, (len(data), 1)) for data in clustered_data]

    # In pg > 1.0 all distributions are multivariate
    pg_dist_type = RobustNormal

    if n_components > 1:
        for cluster in clustered_data:
            if len(cluster) < n_components:
                raise ValueError(
                    f"The training labels did only provide a small number of samples ({len(cluster)}) for one of the "
                    "states. "
                    f"To initialize {n_components} components in a mixture model, we need at least {n_components} "
                    f"samples! "
                    "Ensure that the training data contains enough samples."
                )
        # calculate Mixture Model for each state, clustered by labels
        distributions = []
        for cluster in clustered_data:
            dists = [pg_dist_type() for _ in range(n_components)]
            # With "first-k" init, we don't have any randomnes! -> No need for any random seed
            dist = GeneralMixtureModel(dists, init="first-k", verbose=verbose)
            try:
                dist.fit(cluster)
                distributions.append(dist)
            except:
                # TODO: Not sure if this is a good way to handle this
                # We just append the unfit dist
                print("Fit failed")
                distributions.append(dist)

    else:
        # if n components is just 1 we do not need a mixture model and just build either multivariate Normal
        # Distribution
        distributions = [pg_dist_type().fit(dataset) for dataset in clustered_data]

    return distributions, clustered_data


def extract_transitions_starts_stops_from_hidden_state_sequence(
    hidden_state_sequence: List[np.ndarray],
) -> Tuple[Set[Tuple[int, int]], np.ndarray, np.ndarray]:
    """Extract transitions from hidden state sequence.

    This function will return a list of transitions as well as start and stop labels that can be found within the
    input sequences.

    input = [[1,1,1,1,1,3,3,3,3,2,2,2,2,4,4,4,4,5,5],
             [0,0,1,1,1,3,3,3,3,2,2,2,6]]
    output_transitions = [[s1,s3],
                          [s3,s2],
                          [s2,s4],
                          [s4,s5],
                          [s0,s1],
                          [s2,s6]]

    output_starts = [1,0]
    output_stops = [5,6]
    """
    assert isinstance(hidden_state_sequence, list), "Hidden state sequence must be list!"

    transitions = []
    starts = []
    ends = []
    for labels in hidden_state_sequence:
        starts.append(labels[0])
        ends.append(labels[-1])
        for idx in np.where(abs(np.diff(labels)) > 0)[0]:
            transitions.append((int(labels[idx]), int(labels[idx + 1])))

    if len(transitions) > 0:
        transitions = set(transitions)
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
    >>> create_equidistant_label_sequence(n_labels = 10, n_states = 5)
    array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    >>> create_equidistant_label_sequence(n_labels = 10, n_states = 4)
    array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3])
    >>> create_equidistant_label_sequence(n_labels = 10, n_states = 3)
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
        label_sequence,
        np.repeat(np.arange(n_states - remainder) + max_handled_state, save_repeats),
    )
    max_handled_state = n_states - remainder + max_handled_state
    label_sequence = np.append(
        label_sequence,
        np.repeat(
            np.arange(n_states - max_handled_state) + max_handled_state,
            save_repeats + 1,
        ),
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
        [(s, e) for s, e in zip(transition_starts, transition_ends) if e - s > 0],
        columns=["start", "end"],
    )


def get_train_data_sequences_transitions(
    data_train_sequence: List[SingleSensorData],
    stride_list_sequence: List[SingleSensorStrideList],
    n_states: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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
    data_train_sequence: List[SingleSensorData],
    stride_list_sequence: List[SingleSensorStrideList],
    n_states: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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
    model: Optional[pgHMM],
    data: pd.DataFrame,
    *,
    expected_columns: Tuple[str, ...],
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

    if len(data) < (n_states := len(model.distributions) - 2):
        raise _DataToShortError(
            "The provided feature data is expected to have at least as many samples as the number of states "
            f"of the model ({n_states}). "
            f"But it only has {len(data)} samples."
        )

    data = np.ascontiguousarray(data.to_numpy())
    try:
        labels_predicted = np.asarray(model.predict([data.copy()]))
    except Exception as e:  # noqa: broad-except
        # TODO: Decide if check still necessary
        # if not model_params_are_finite(model):
        #     raise ValueError(
        #         "Prediction failed! (See error above.). "
        #         "However, the provided pomegranate model has non-finite/NaN parameters. "
        #         "This might be the source of the observed error and indicates problems during training. "
        #         "Check the training history and the model parameters to confirm invalid training behaviour. "
        #         "Unfortunately, there is no way to automatically fix these issues. "
        #         "Simply speaking, your training data could not be represented well by the selected model architecture. "
        #         "Check for obvious errors in your pre-processing or try to use a different model architecture. "
        #     ) from e
        raise ValueError(
            "Prediction failed! (See error above.). "
            "Unfortunately, we are not sure what happened. "
            "The error was caused by pomegrante internals."
        ) from e

    return np.asarray(labels_predicted)


def labels_to_prior(labels_sequence, n_states, certainty=0.9):
    labels_sequence_as_prior = []
    per_state_prior = []
    for state in range(n_states):
        tmp = np.zeros(n_states)
        tmp[state] = certainty
        tmp[tmp == 0] = 1 - certainty
        per_state_prior.append(tmp)
    per_state_prior = np.asarray(per_state_prior)
    # normalize
    per_state_prior = per_state_prior / per_state_prior.sum(axis=1)[:, np.newaxis]

    for labels in labels_sequence:
        labels_sequence_as_prior.append(per_state_prior[labels.astype(int)])

    return labels_sequence_as_prior


def freeze_nested_distribution(distribution):
    distribution.frozen = torch.tensor(True)
    if nested_dists := getattr(distribution, "distributions", None):
        for nested_dist in nested_dists:
            freeze_nested_distribution(nested_dist)
