import matplotlib.pyplot as plt
import numpy as np
import pomegranate as pg

from gaitmap.utils.array_handling import bool_array_to_start_end_array, start_end_array_to_bool_array

N_JOBS = 1


def create_transition_matrix_fully_connected(n_states):
    """Create nxn transition matrix with only 1 entries"""

    transition_matrix = np.ones((n_states, n_states)) / n_states
    start_probs = np.ones(n_states)
    end_probs = np.ones(n_states)

    return [transition_matrix, start_probs, end_probs]


def create_transition_matrix_left_right(n_states, self_transition=True):
    """Create nxn transition for left to right model"""
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

    return [transition_matrix, start_probs, end_probs]


def print_transition_matrix(model, precision=3):
    """Print model transition matrix in user friendly format"""
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision)
    if isinstance(model, pg.HiddenMarkovModel):
        print(model.dense_transition_matrix()[0:-2, 0:-2])
    if isinstance(model, np.ndarray):
        print(model)


def print_model_distribution(model):
    for state in model.states:
        if state.distribution != None:
            print(
                "%s: %.3f ± %.3f"
                % (
                    state.name,
                    state.distribution.parameters[0],
                    state.distribution.parameters[1],
                )
            )


def cluster_data_by_labels(data_list, label_list, debug_plot=False):
    """Cluster data by labels"""
    # just check everything for correct types
    if not isinstance(label_list, list):
        label_list = [label_list]

    label_list = [np.asarray(l).tolist() for l in label_list]

    if not isinstance(data_list, list):
        data_list = [label_list]

    data_list = [np.asarray(d).tolist() for d in data_list]

    X_ = [x for x, label in zip(data_list, label_list) if label != None]  # remove datasets where the labellist is None
    X_ = np.concatenate(X_)  # concatenate all datasets with label_list to a single array

    labels_ = np.concatenate(
        [l for l in label_list if l is not None]
    )  # concatenate all not None labellists to a single array
    label_set = np.unique(labels_)  # get set of unique label_list

    clustered_data = [X_[labels_ == label] for label in label_set]

    if debug_plot:
        fig, ax = plt.subplots(len(clustered_data), 1, sharex="all", sharey="all")
        for i, data in enumerate(clustered_data, start=0):
            ax[i].plot(data)
            ax[i].set_title("s" + str(i))

    return clustered_data


def gmms_from_samples(data, labels, n_components, random_seed=None, verbose=False, debug_plot=False):
    """This function clusteres the data by the given labels and fits either univariate or multivariate
    Normal Distributions for each cluster. If n_components is > 1 then a Mixture Model of n-univariate or n-multivariate
    Gaussian Distributions will be fitted."""

    if not np.array(data).flags["C_CONTIGUOUS"]:
        raise ValueError("Memory Layout of given input data is not contiguous! Consider using ")

    clustered_data = cluster_data_by_labels(data, labels, debug_plot)

    if random_seed:
        np.random.seed(random_seed)

    # select correct distribution type based on dimensionality of the given data
    if data[0].ndim == 1:
        pg_dist_type = pg.NormalDistribution
        # we need to reshape the 1D-data for pomegranate!
        clustered_data = [np.reshape(data, (len(data), 1)) for data in clustered_data]
    else:
        pg_dist_type = pg.MultivariateGaussianDistribution

    if n_components > 1:
        distributions = [
            pg.GeneralMixtureModel.from_samples(
                pg_dist_type, n_components=n_components, X=dataset, verbose=verbose, n_jobs=N_JOBS
            )
            for dataset in clustered_data
        ]  # calculate Mixture Model for each state, clustered by labels
    else:
        # if n components is just 1 we do not need a mixture model and just build either univariate or multivariate
        # Normal Distributiion depending on the input data dimension
        distributions = [pg_dist_type.from_samples(dataset) for dataset in clustered_data]

    return [distributions, clustered_data]


def norm_dist_from_samples(data, labels, random_seed=None, verbose=False, debug_plot=False):
    clustered_data = cluster_data_by_labels(data, labels, debug_plot)

    distributions = []
    if random_seed:
        np.random.seed(random_seed)
    distributions = [
        pg.NormalDistribution.from_samples(dataset) for dataset in clustered_data
    ]  # calculate Mixture Model for each state, clustered by labels
    return [distributions, clustered_data]


def fix_model_names(model):
    """Replace state name from s10 to sN with characters as pomegranate seems to have a "sorting" bug. Where states
    get sorted like s0, s1, s10, s2, .... so we will map state names >10 to letters. E.g. "s10" -> "sa", "s11" -> "sb"
     usw."""
    for state in model.states:
        if state.name[0] == "s":
            state_number = int(state.name[1:])
            # replace state numbers >= 10 by characters form the ascii-table :)
            if state_number >= 10:
                state.name = "s" + chr(87 + state_number)
    return model


def get_state_by_name(model, state_name):
    """Get state object from model by name."""
    for state in model.states:
        if state.name == state_name:
            return state
    raise ValueError("State %s not found within given model!" % state_name)


def add_transition(model, transition, transition_probability):
    """Add a transition to an existing model by state-names e.g.

    add_transition(model, transition = ["s0","s1"], transition_probability = 0.5)

    to add a edge from state s0 to state s1 with a transition probability of 0.5
    """

    model.add_transition(
        get_state_by_name(model, transition[0]),
        get_state_by_name(model, transition[1]),
        transition_probability,
    )


def predict(model, data, algorithm="viterbi"):
    """Perform prediction based on given data and given model."""
    # need to check if memory layout of given data is
    # see related pomegranate issue: https://github.com/jmschrei/pomegranate/issues/717
    if not np.array(data).flags["C_CONTIGUOUS"]:
        raise ValueError("Memory Layout of given input data is not contiguois! Consider using ")

    labels_predicted = np.asarray(model.predict(data, algorithm=algorithm))
    # pomegranate always adds an additional label for the start- and end-state, which can be ignored here!
    return np.asarray(labels_predicted[1:-1])


def get_model_distributions(model):
    """Returns all not None distributions as list from given model."""
    distributions = []
    for state in model.states:
        if state.distribution != None:
            distributions.append(state.distribution)
    return distributions


def labels_to_strings(labelsequence):
    """Convert label sequence of ints to strings. Why do we need this?
    Pomegranated messes up sorting of states: it will sort like this: s0, s1, s10, s2.... which can lead to unexpected
    behaviour."""
    if not isinstance(labelsequence, list):
        labelsequence = [labelsequence]

    labelsequence_str = []
    for sequence in labelsequence:
        if isinstance(sequence, np.ndarray):
            sequence = sequence.tolist()
        if sequence == None:
            labelsequence_str.append(sequence)
            continue
        labels = np.asarray(sequence)
        labels_str = labels.astype(str).copy()
        for i in np.unique(labels).astype(int):

            if i >= 10:
                labels_str[labels == i] = "s" + chr(87 + i)
            else:
                labels_str[labels == i] = "s" + str(int(i))
        labels_str[labels_str == "nan"] = None
        # labelsequence_str.append(labels_str.tolist())
        labelsequence_str.append(np.asarray(labels_str))
    return labelsequence_str


def extract_transitions_starts_stops_from_hidden_state_sequence(hidden_state_sequence):
    """This function will return a list of transitions as well as start and stop labels that can be found within the
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
    if not isinstance(hidden_state_sequence, list):
        hidden_state_sequence = [hidden_state_sequence]

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

    return [transitions, starts, ends]


def labels_to_strings(labelsequence):
    """Convert label sequence of ints to strings. Why do we need this?
    Pomegranated messes up sorting of states: it will sort like this: s0, s1, s10, s2.... which can lead to unexpected
    behaviour."""
    if not isinstance(labelsequence, list):
        labelsequence = [labelsequence]

    labelsequence_str = []
    for sequence in labelsequence:
        if isinstance(sequence, np.ndarray):
            sequence = sequence.tolist()
        if sequence == None:
            labelsequence_str.append(sequence)
            continue
        labels = np.asarray(sequence)
        labels_str = labels.astype(str).copy()
        for i in np.unique(labels).astype(int):

            if i >= 10:
                labels_str[labels == i] = "s" + chr(87 + i)
            else:
                labels_str[labels == i] = "s" + str(int(i))
        labels_str[labels_str == "nan"] = None
        # labelsequence_str.append(labels_str.tolist())
        labelsequence_str.append(np.asarray(labels_str))
    return labelsequence_str


def create_equidistant_labels_from_label_list(data, label_list, n_states, state_offset=1):
    """This function takes a single data sequence e.g. one gait bout and a corresponding label_list with "start-end"
    values as input. Retrun value will be labels for the given dataset with stair like ascending labels for each stride
    with equidistance step width."""
    labels = np.zeros(len(data))

    for label in label_list:
        l = label[1] - label[0]
        n_samples = int(round(l / n_states))

        for j in np.arange(0, n_states):
            start = label[0] + j * n_samples
            labels[start : start + n_samples] = j + state_offset
        labels[start + n_samples : label[1]] = j + state_offset

    return labels


def create_equidistant_label_sequence(n_labels, n_states):
    """create label sequence of length n_states with n_labels unique labels. This can be used to e.g. initialize labels
    for a single stride."""
    # calculate the samples per state (must be integer!)
    n_labels_per_state = int(round(n_labels / n_states))

    label_sequence = np.zeros(n_labels)

    for state in np.arange(0, n_states):
        start = state * n_labels_per_state
        label_sequence[start : start + n_labels_per_state] = state

    # fill remaining array with last state
    label_sequence[start + n_labels_per_state :] = state

    return label_sequence


def get_train_data_sequences_transitions(data_train_sequence, stride_list_sequence, n_states):
    """Extract Transition Training set.

    - data_train_sequence: list of datasets in feature space

    - stride_list_sequence: list of gaitmap stride-lists

     - n_states: number of labels

    """

    # TODO: check that we have a valid list of input sequences
    # TODO: check that we have a list of valid stride sequences

    transition_data_train_sequence = []
    transition_labels_train_sequence = []

    # iterate over all given training sequences, extract all strides and generate equidistant labels for each of these strides.
    for data, stride_list in zip(data_train_sequence, stride_list_sequence):

        # here we will only extract transitions from the given bout (aka everything which is "not marked as a stride")
        transition_mask = np.invert(
            start_end_array_to_bool_array(stride_list[["start", "end"]].to_numpy(), pad_to_length=len(data) - 1).astype(
                bool
            )
        )
        transition_start_end_list = bool_array_to_start_end_array(transition_mask)

        # for each transition, get data and create some naive labels for initialization
        for start, end in transition_start_end_list:
            # append extracted sequences and corresponding label set to resuls list
            transition_data_train_sequence.append(data[start : end + 1])
            transition_labels_train_sequence.append(
                create_equidistant_label_sequence(end - start + 1, n_states).astype(int)
            )

    return transition_data_train_sequence, transition_labels_train_sequence


def get_train_data_sequences_strides(data_train_sequence, stride_list_sequence, n_states):
    """Extract Transition Training set.


    - data_train_sequence: list of datasets in feature space

    - stride_list_sequence: list of gaitmap stride-lists

    - n_states: number of labels

    """

    # TODO: check that we have a valid list of input sequences
    # TODO: check that we have a list of valid stride sequences

    stride_data_train_sequence = []
    stride_labels_train_sequence = []

    for data, stride_list in zip(data_train_sequence, stride_list_sequence):
        # extract strides directly from stride_list
        for start, end in stride_list[["start", "end"]].to_numpy():
            stride_data_train_sequence.append(data[start : end + 1])
            stride_labels_train_sequence.append(
                create_equidistant_label_sequence(end - start + 1, n_states).astype(int)
            )

    return stride_data_train_sequence, stride_labels_train_sequence
