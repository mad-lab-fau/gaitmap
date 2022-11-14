"""Utils and helper functions for HMM classes."""
import json
from typing import Any, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import pomegranate as pg
from tpcp import BaseTpcpObject
from tpcp._hash import custom_hash

from gaitmap.utils.array_handling import bool_array_to_start_end_array, start_end_array_to_bool_array
from gaitmap.utils.datatype_helper import SingleSensorData, SingleSensorStrideList


def _add_transition(model, a, b, probability, pseudocount, group):
    """Hacky way to add a transition when cloning a model in the "wrong" way."""
    pseudocount = pseudocount or probability
    model.graph.add_edge(a, b, probability=probability, pseudocount=pseudocount, group=group)


def _clone_model(orig_model: pg.HiddenMarkovModel, assert_correct: bool = True) -> pg.HiddenMarkovModel:
    """Clone a HMM without changing its values using a hacky way.

    XXX: This method can clone a HMM by copying over all values individually.
    It skips the init of the models, as using them lead to rounding issues, that result in models that are not
    strictly identical.
    This should not be a problem mathematically (as the rounding errors are small), but if you expect to get an
    exact copy, this method can help you.

    This should only be used internally! If you have a tpcp/gaitmap algorithm that contains a HMM as input param,
    you should use `_clone_model` in the implementation, when you need a copy.
    To make your algorithm itself properly clonable (including your HMM model), the algorithm needs to inherit from
    `_HackyClonableHMMFix`.
    This will overwrite how your algorithm handles `tpcp.clone`.
    Note, it will not fix how deepcopy (`copy.deepcopy`) is handled.

    .. warning:: This will not work all the time! In particular when transitions are added to the model after the
                 initial creation, it seems like it is impossible to clone the model correctly, because the order of
                 the edges can not be restored.
                 A possible workaround in this case is to clone the model twice.
                 The first clone will reorder the edges in a predictable way and the order will stay consistent
                 afterwards.

    Parameters
    ----------
    orig_model
        The HMM model to clone
    assert_correct
        If True, the cloned model will be compared to the original model and an AssertionError will be raised if they
        not identical.
        In general, this should be True, however, when you want to clone a model with certain edges that are not
        fully covered, you can set this to False at your own risk.

    Returns
    -------
    model
        A deepcopy of the HMM model.

    """
    d = json.loads(orig_model.to_json())

    # Make a new generic HMM
    model = pg.HiddenMarkovModel(str(d["name"]))

    with np.errstate(divide="ignore"):
        states = [pg.State.from_dict(j) for j in d["states"]]

    for cloned_state, state in zip(states, orig_model.states):
        assert cloned_state.name == state.name
        if isinstance(state.distribution, pg.GeneralMixtureModel):
            # Fix the distribution weights
            # Note the `[:]`! This is important, because pg keeps a pointer to the original weights vector internally.
            # If we would reassign weights (and not just its content), we would update weights, but pg would still
            # use the old values internally, as it uses the pointer to access it.
            cloned_state.distribution.weights[:] = np.copy(state.distribution.weights)

    for i, j in d["distribution ties"]:
        # Tie appropriate states together
        states[i].tie(states[j])

    # Add all the states to the model
    model.add_states(states)

    # Indicate appropriate start and end states
    model.start = states[d["start_index"]]
    model.end = states[d["end_index"]]

    new_state_order = [state.name for state in states]

    # Add all the edges to the model
    for (start, end, data) in list(orig_model.graph.edges(data=True)):
        _add_transition(
            model,
            states[new_state_order.index(start.name)],
            states[new_state_order.index(end.name)],
            data["probability"],
            data["pseudocount"],
            data["group"],
        )

    # Bake the model
    model.bake(verbose=False)

    if assert_correct:
        assert custom_hash(model) == custom_hash(
            orig_model
        ), "Cloning the provided HMM model failed! Please open an issue on github with an example."

    return model


class _HackyClonableHMMFix(BaseTpcpObject):
    """A hacky implementation to ensure that HMM parameters are actually cloned, when cloning the algorithm.

    This implements and alternative cloning method for all parameters that are of type `pg.HiddenMarkovModel` that is
    used when `tpcp.clone` is used.
    When you implement an algorithm with a `pg.HiddenMarkovModel` as parameter, you should inherit from this class.
    In addition, you should use `_clone_model` internally, when you need a identical copy/clone of your hmm.

    For more information see the `_clone_model` function.

    """

    @classmethod
    def __clone_param__(cls, param_name: str, value: Any) -> Any:
        """Overwrite cloning for HMM models.

        XXX: This is hacky shit and it is stupid that I have to do it in the first place, but there is no build in
        way in pomegrante to properly deepcopy a HMM.
        For several reasons, a deepcopied HMM will only be approximately identical to the original (rounding issues).
        Our cloning implementation, does some bad stuff (and should always be covered by tests), but seems to
        properly clone the objects (so that the hashs are identical).
        """
        if isinstance(value, pg.HiddenMarkovModel):
            return _clone_model(value)
        return super().__clone_param__(param_name, value)


def create_transition_matrix_fully_connected(n_states: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create nxn transition matrix with only 1 entries."""
    transition_matrix = np.ones((n_states, n_states)) / n_states
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


def print_transition_matrix(model: pg.HiddenMarkovModel, precision: int = 3):
    """Print model transition matrix in user-friendly format."""
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision)
    if isinstance(model, pg.HiddenMarkovModel):
        print(model.dense_transition_matrix()[0:-2, 0:-2])
    if isinstance(model, np.ndarray):
        print(model)


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

    clustered_data = [x_[labels_ == label] for label in label_set]

    return clustered_data


def gmms_from_samples(
    data,
    labels,
    n_components: int,
    verbose: bool = False,
    n_init: int = 5,
    n_jobs: int = 1,
):
    """Create Gaussian Mixture Models from samples.

    This function clusters the data by the given labels and fits either univariate or multivariate
    Normal Distributions for each cluster. If n_components is > 1 then a Mixture Model of n-univariate or n-multivariate
    Gaussian Distributions will be fitted.
    """
    if not np.array(data, dtype=object).data.c_contiguous:
        raise ValueError("Memory Layout of given input data is not contiguous! Consider using numpy.ascontiguousarray.")

    clustered_data = cluster_data_by_labels(data, labels)

    # select correct distribution type based on dimensionality of the given data
    if data[0].ndim == 1:
        pg_dist_type = pg.NormalDistribution
        # we need to reshape the 1D-data for pomegranate!
        clustered_data = [np.reshape(data, (len(data), 1)) for data in clustered_data]
    else:
        pg_dist_type = pg.MultivariateGaussianDistribution

    if n_components > 1:
        # calculate Mixture Model for each state, clustered by labels
        distributions = [
            pg.GeneralMixtureModel.from_samples(
                pg_dist_type,
                n_components=n_components,
                X=dataset,
                verbose=verbose,
                n_jobs=n_jobs,
                n_init=n_init,
                init="first-k",  # With this initialisation, we don't have any randomnes! -> No need for any random
                # seed.
            )
            for dataset in clustered_data
        ]
    else:
        # if n components is just 1 we do not need a mixture model and just build either univariate or multivariate
        # Normal Distribution depending on the input data dimension
        distributions = [pg_dist_type.from_samples(dataset) for dataset in clustered_data]

    return distributions, clustered_data


def norm_dist_from_samples(data, labels, random_seed=None):
    """Create Normal Distributions from samples."""
    clustered_data = cluster_data_by_labels(data, labels)

    if random_seed:
        np.random.seed(random_seed)
    distributions = [
        pg.NormalDistribution.from_samples(dataset) for dataset in clustered_data
    ]  # calculate Mixture Model for each state, clustered by labels
    return distributions, clustered_data


def fix_model_names(model):
    """Fix pomegranate model names.

    Replace state name from s10 to sN with characters as pomegranate seems to have a "sorting" bug. Where states
    get sorted like s0, s1, s10, s2, .... so we will map state names >10 to letters. E.g. "s10" -> "sa", "s11" -> "sb"
    """
    for state in model.states:
        if state.name[0] == "s":
            state_number = int(state.name[1:])
            # replace state numbers >= 10 by characters form the ascii-table :)
            if state_number >= 10:
                state.name = "s" + chr(87 + state_number)
    return model


def get_state_by_name(model: pg.HiddenMarkovModel, state_name: str) -> str:
    """Get state object from model by name."""
    for state in model.states:
        if state.name == state_name:
            return state
    raise ValueError(f"State {state_name} not found within given _model!")


def add_transition(model: pg.HiddenMarkovModel, transition: Tuple[str, str], transition_probability: float):
    """Add a transition to an existing model by state-names.

    add_transition(model, transition = ("s0","s1"), transition_probability = 0.5)
    to add a edge from state s0 to state s1 with a transition probability of 0.5.
    """
    model.add_transition(
        get_state_by_name(model, transition[0]),
        get_state_by_name(model, transition[1]),
        transition_probability,
    )


def get_model_distributions(model: pg.HiddenMarkovModel) -> List[pg.Distribution]:
    """Return all not None distributions as list from given model."""
    distributions = []
    for state in model.states:
        if state.distribution is not None:
            distributions.append(state.distribution)
    return distributions


def labels_to_strings(labelsequence: List[Optional[np.ndarray]]) -> List[Optional[List[str]]]:
    """Convert label sequence of ints to strings.

    Pomegranated messes up sorting of states: it will sort like this: s0, s1, s10, s2.... which can lead to unexpected
    behaviour.
    """
    assert isinstance(labelsequence, list), "labelsequence must be list!"

    labelsequence_str = []
    for sequence in labelsequence:
        if sequence is None:
            labelsequence_str.append(sequence)
            continue
        labelsequence_str.append([f"s{i:02}" for i in sequence])
    return labelsequence_str


def extract_transitions_starts_stops_from_hidden_state_sequence(
    hidden_state_sequence: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract transitions from hidden state sequence.

    This function will return a list of transitions as well as start and stop labels that can be found within the
    input sequences.

    input = [[1,1,1,1,1,3,3,3,3,2,2,2,2,4,4,4,4,5,5],
             [0,0,1,1,1,3,3,3,3,2,2,2,6]]
    output_transitions = [[1,3],
                          [3,2],
                          [2,4],
                          [4,5],
                          [0,1],
                          [2,6]]

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
            transitions.append([labels[idx], labels[idx + 1]])

    if len(transitions) > 0:
        transitions = np.unique(transitions, axis=0).astype(int)
    starts = np.unique(starts).astype(int)
    ends = np.unique(ends).astype(int)

    return transitions, starts, ends


def create_equidistant_label_sequence(n_labels: int, n_states: int) -> np.ndarray:
    """Create equidistant label sequence.

    create label sequence of length n_states with n_labels unique labels. This can be used to e.g. initialize labels
    for a single stride.
    """
    # calculate the samples per state (must be integer!)
    n_labels_per_state = int(round(n_labels / n_states))

    label_sequence = np.zeros(n_labels)

    state = None
    if n_states < 1:
        raise ValueError("Number of states must be at least 1!")

    end = 0
    for state in range(0, n_states):
        start = state * n_labels_per_state
        end = start + n_labels_per_state
        label_sequence[start:end] = state
    # fill remaining array with last state
    label_sequence[end:] = state

    return label_sequence


def get_train_data_sequences_transitions(
    data_train_sequence: List[SingleSensorData], stride_list_sequence: List[SingleSensorStrideList], n_states: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Extract Transition Training set.

    - data_train_sequence: list of datasets in feature space
    - stride_list_sequence: list of gaitmap stride-lists
    - n_states: number of labels.
    """
    trans_data_train_sequence = []
    trans_labels_train_sequence = []

    # iterate over all given training sequences, extract all strides and generate equidistant labels for each of these
    # strides.
    for data, stride_list in zip(data_train_sequence, stride_list_sequence):

        # here we will only extract transitions from the given bout (aka everything which is "not marked as a stride")
        transition_mask = np.invert(
            start_end_array_to_bool_array(stride_list[["start", "end"]].to_numpy(), pad_to_length=len(data) - 1).astype(
                bool
            )
        )
        transition_start_end_list = bool_array_to_start_end_array(transition_mask)
        # fix last indices
        transition_start_end_list[-1][-1] = transition_start_end_list[-1][-1] + 1

        # for each transition, get data and create some naive labels for initialization
        for start, end in transition_start_end_list:
            # append extracted sequences and corresponding label set to resuls list
            trans_data_train_sequence.append(data[start:end])
            trans_labels_train_sequence.append(create_equidistant_label_sequence(end - start, n_states).astype(int))

    return trans_data_train_sequence, trans_labels_train_sequence


def get_train_data_sequences_strides(
    data_train_sequence: List[SingleSensorData], stride_list_sequence: List[SingleSensorStrideList], n_states: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Extract Transition Training set.

    - data_train_sequence: list of datasets in feature space
    - stride_list_sequence: list of gaitmap stride-lists
    - n_states: number of labels.
    """
    stride_data_train_sequence = []
    stride_labels_train_sequence = []

    for data, stride_list in zip(data_train_sequence, stride_list_sequence):
        # extract strides directly from stride_list
        for start, end in stride_list[["start", "end"]].to_numpy():
            stride_data_train_sequence.append(data[start:end])
            stride_labels_train_sequence.append(create_equidistant_label_sequence(end - start, n_states).astype(int))

    return stride_data_train_sequence, stride_labels_train_sequence


def predict(
    model: Optional[pg.HiddenMarkovModel],
    data: pd.DataFrame,
    *,
    expected_columns: Tuple[str, ...],
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

    data = np.ascontiguousarray(data.to_numpy())
    labels_predicted = np.asarray(model.predict(data.copy(), algorithm=algorithm))
    # pomegranate always adds an additional label for the start- and end-state, which can be ignored here!
    return np.asarray(labels_predicted[1:-1])
