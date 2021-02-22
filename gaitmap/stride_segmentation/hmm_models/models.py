"""Dtw template base classes and helper."""
from importlib.resources import open_text
from typing import Optional, List
import json
from pomegranate import HiddenMarkovModel as pgHMM
import pomegranate as pg
import numpy as np
from gaitmap.base import _BaseSerializable
from gaitmap.stride_segmentation.hmm_models.utils import (
    gmms_from_samples,
    create_transition_matrix_left_right,
    create_transition_matrix_fully_connected,
    fix_model_names,
    get_model_distributions,
    predict,
    extract_transitions_starts_stops_from_hidden_state_sequence,
    add_transition,
    labels_to_strings,
    create_equidistant_labels_from_label_list,
    create_equidistant_label_sequence,
    get_train_data_sequences_transitions,
    get_train_data_sequences_strides,
)
import copy

VERBOSE_MODEL = True
VERBOSE_DISTRIBUTIONS = False
DEBUG_PLOTS = False
N_JOBS = 1


def initialize_hmm(
    data_train_sequence,
    labels_initialization_sequence,
    n_states,
    n_gmm_components,
    architecture,
    random_seed=None,
    name="untrained",
):
    """Model Initialization.

    - data_train_sequence (list of np.ndarrays):
        list of training sequences this might be e.g. a list of strides where each strides is represented by one
        np.ndarray (which might contain multiple dimensions)

    - labels_initialization_sequence (list of np.ndarrays):
        list of labels which are used to initialize the emission distributions. Note: These labels are just for
        initialization. Distributions will be optimized later in the training process of the model!
        Length of each np.ndarray in "labels_initialization_sequence" needs to match the length of each
        corresponding np.ndarray in "data_train_sequence"

    - n_gmm_components (integer) : of components for multivariate distributions
    - architecture     (str)     : type of model architecture, for more details see below
    - n_states         (integer) : number of hidden-states within the model
    - random_seed      (float)   : fix seed for random number generation to make sure that results will be
                                       reproducible. Set to None if not used.
                                       TODO: check if random_seed is actually only used if we use k-means somewhere...

    This function supports currently the following "architectures":
        - "left-right-strict":
        This will result in a strictly left-right structure, with no self-transitions and
        start- and end-state bound to the first and last state, respectively.
        Example transition matrix for a 5-state model:
        transition_matrix: 1  1  0  0  0   starts: 1  0  0  0  0
                           0  1  1  0  0   stops:  0  0  0  0  1
                           0  0  1  1  0
                           0  0  0  1  1
                           0  0  0  0  1

        - "left-right-loose":
        This will result in a loose left-right structure, with allowed self-transitions and
        start- and end-state not specified initially.
        Example transition matrix for a 5-state model:
        transition_matrix: 1  1  0  0  0   starts: 1  1  1  1  1
                           0  1  1  0  0   stops:  1  1  1  1  1
                           0  0  1  1  0
                           0  0  0  1  1
                           1  0  0  0  1

         - "fully-connected":
        This will result in a fully connected structure where all existing edges are initialized with the same probability.
        Example transition matrix for a 5-state model:
        transition_matrix: 1  1  1  1  1   starts: 1  1  1  1  1
                           1  1  1  1  1   stops:  1  1  1  1  1
                           1  1  1  1  1
                           1  1  1  1  1
                           1  1  1  1  1
    """

    [distributions, _] = gmms_from_samples(
        data_train_sequence,
        labels_initialization_sequence,
        n_gmm_components,
        # make sure thate "None" string gets converted to proper None
        random_seed=random_seed,
        verbose=VERBOSE_DISTRIBUTIONS,
        debug_plot=DEBUG_PLOTS,
    )

    # if we force the model into a left-right architecture we know that stride borders should correspond to the point where the model "loops" (aka state-0 and state-n)
    # so we also will enforce the model to start with "state-0" and always end with "state-n"
    if architecture == "left-right-strict":
        [transition_matrix, start_probs, end_probs] = create_transition_matrix_left_right(
            n_states, self_transition=False
        )

    # allow transition model to start and end in all states (as we do not have any specific information about "transitions", this could be actually anything in the data which is no stride)
    if architecture == "left-right-loose":
        [transition_matrix, _, _] = create_transition_matrix_left_right(n_states, self_transition=True)

        start_probs = np.ones(n_states).astype(float)
        end_probs = np.ones(n_states).astype(float)

    # fully connected model with all transitions initialized equally. Allowing all possible transitions.
    if architecture == "fully-connected":
        [transition_matrix, start_probs, end_probs] = create_transition_matrix_fully_connected(n_states)

    model = pg.HiddenMarkovModel.from_matrix(
        transition_probabilities=transition_matrix,
        distributions=copy.deepcopy(distributions),
        starts=start_probs,
        ends=end_probs,
        verbose=VERBOSE_MODEL,
    )

    # pomegranate seems to have a strange sorting bug where state names >= 10 (e.g. s10 get sorted in a bad order like s0, s1, s10, s2 usw..)
    model = fix_model_names(model)
    # make sure that transition-matrix is normalized
    model.bake()
    model.name = name

    return model


def train_hmm(model_untrained, data_train_sequence, max_iterations, stop_threshold, algo_train, name="trained"):
    """Model Training

    - model_untrained (pomegranate.HiddenMarkovModel):
        pomegranate HiddenMarkovModel object with initialized distributions and transition matrix

    - data_train_sequence (list of np.ndarrays):
        list of training sequences this might be e.g. a list of strides where each strides is represented by one
        np.ndarray (which might contain multiple dimensions)

    - algo_train (str):
        algorithm for training, can be "viterbi", "baum-welch" or "labeled"

    - stop_threshold (float):
        termination criteria for training improvement e.g. 1e-9

    - max_iterations (int):
        termination criteria for training iteration number e.g. 1000

    """

    # check if all training sequences have a minimum length of n-states, smaller sequences or empty sequences can lead to unexpected behaviour!
    length_of_training_sequences = np.array([len(data) for data in data_train_sequence])
    if np.any(length_of_training_sequences < (len(model_untrained.states) - 2)):
        raise ValueError(
            "Length of all training sequences must be equal or larger than the number of states in the given model!"
        )

    # make copy from untrained model, as pomegranate will just update parameters in the given model and not returning a copy
    model_trained = copy.deepcopy(model_untrained)

    _, history = model_trained.fit(
        sequences=np.array(data_train_sequence).copy(),
        labels=None,
        algorithm=algo_train,
        stop_threshold=stop_threshold,
        max_iterations=max_iterations,
        return_history=True,
        verbose=VERBOSE_MODEL,
        n_jobs=N_JOBS,
    )
    model_trained.name = name

    return model_trained, history


def build_and_train_sub_hmm(
    data_train_sequence_list,
    initial_hidden_states_sequence_list,
    n_states,
    n_gmm_components,
    algo_train,
    stop_threshold,
    max_iterations,
    random_seed,
    architecture,
    model_name,
):
    """Create a single semi-supervised trained HMM model.

    :param data_train_sequence:
        This is should be a list of training sequences e.g. a list of single stride sequences aka a list of np.ndarrays
    :param borders_train_sequence:
        This is should be a list of initial hidden state sequences, corresponding to the given data_train_sequences list
        the length of both input elements must be the same (as well as the length of individual entry of the
        data_train_sequence_list, must have a matching length sequences within th initial_hidden_states_sequence_list.
    :param n_states:
    :param n_gmm_components:
    :param algo_train:
    :param stop_threshold:
    :param max_iterations:
    :param random_seed:
    :param architecture:
    :param model_name:

    :return:
    """

    if len(data_train_sequence_list) != len(initial_hidden_states_sequence_list):
        raise ValueError(
            "The given training sequence and initial training labels do not match in their number of individual sequences!"
            "len(data_train_sequence_list) = %d !=  %d = len(initial_hidden_states_sequence_list)"
            % (len(data_train_sequence_list), len(initial_hidden_states_sequence_list))
        )

    # initialize model by naive equidistant labels
    stride_model_untrained = initialize_hmm(
        data_train_sequence_list,
        initial_hidden_states_sequence_list,
        n_states,
        n_gmm_components,
        architecture,
        random_seed=random_seed,
        name=model_name + "-untrained",
    )
    # train model
    stride_model_trained, history = train_hmm(
        stride_model_untrained,
        data_train_sequence_list,
        max_iterations,
        stop_threshold,
        algo_train,
        name=model_name + "-trained",
    )
    return stride_model_trained


def create_stride_hmm(
    data_train_sequence,
    borders_train_sequence,
    n_states=25,
    n_gmm_components=5,
    algo_train="baum-welch",
    stop_threshold=1e-9,
    max_iterations=100,
    random_seed=None,
    architecture="left-right-strict",
):
    # extract all strides from data and initialize each stride with equidistant labels
    [stride_data_train_sequence, stride_labels_train_sequence] = get_train_data_sequences_strides(
        data_train_sequence, borders_train_sequence, n_states
    )
    # TODO: check for invalid sequence length!

    return build_and_train_sub_hmm(
        stride_data_train_sequence,
        stride_labels_train_sequence,
        n_states,
        n_gmm_components,
        algo_train,
        stop_threshold,
        max_iterations,
        random_seed,
        architecture,
        "stride_model",
    )


def create_transition_hmm(
    data_train_sequence,
    borders_train_sequence,
    n_states=5,
    n_gmm_components=5,
    algo_train="baum-welch",
    stop_threshold=1e-9,
    max_iterations=100,
    random_seed=None,
    architecture="left-right-loose",
):
    # extract all transitions from data and initialize each transition with equidistant labels
    [transition_data_train_sequence, transition_labels_train_sequence] = get_train_data_sequences_transitions(
        data_train_sequence, borders_train_sequence, n_states
    )
    # TODO: check for invalid sequence length!

    return build_and_train_sub_hmm(
        transition_data_train_sequence,
        transition_labels_train_sequence,
        n_states,
        n_gmm_components,
        algo_train,
        stop_threshold,
        max_iterations,
        random_seed,
        architecture,
        "transition_model",
    )


def build_combined_transition_stride_model(
    data_train_sequence,
    borders_train_sequence,
    transition_model,
    stride_model,
    algo_train,
    algo_predict,
    init_method,
    max_iterations,
    stop_threshold,
):
    """For model combination actually only the transition probabilities will be updated, while keeping the already
    learned distributions for all states. This can be achieved by "labeled" training, where basically just the
    number of transitions will be counted."""

    # some initialization stuff...
    n_states_transition = len(transition_model.states) - 2  # subtract silent start- and end-state
    n_states_stride = len(stride_model.states) - 2  # subtract silent start- and end-state
    n_states = n_states_transition + n_states_stride

    """To find the "actual" hidden-state labels for "labeled-training" with the given training data set, we will again 
    split everything into strides and transitions based on our initial stride borders and then predict the labels with 
    the respective already learned models.

    To rephrase it again: We want to create a fully labeled dataset with already optimal hidden-state labels, but as these
    lables are hidden, we need to predict them with our already trained models...
    """
    labels_train_sequence = []

    for data_train, border_list_train in zip(data_train_sequence, borders_train_sequence):
        labels_train = np.zeros(len(data_train))
        if border_list_train.size == 2:
            border_list_train = [border_list_train]
        # predict hidden-state sequence for each stride using "stride model"
        for stride in border_list_train:
            stride_data_train = data_train[stride[0] : stride[1]]
            if len(stride_data_train) >= n_states_stride:
                hidden_state_sequence = predict(
                    stride_model, stride_data_train, algorithm=algo_predict
                )  # predict hidden state labels
                hidden_state_sequence = (
                    hidden_state_sequence + n_states_transition
                )  # add label offset for stride states
                labels_train[stride[0] : stride[1]] = hidden_state_sequence

        # here we will only extract transitions from the given bout
        [transition_start_end_list, _] = label_helper.bin_array_to_sequence_list(
            label_helper.flatten_start_stop_list_to_binary_array(
                border_list_train, pad_to_length=len(data_train)
            ).astype(bool)
        )

        # for each transition, get data and create some naive labels for initialization
        for start_end in transition_start_end_list:
            transition_data_train = data_train[start_end[0] : start_end[1] + 1]
            if len(transition_data_train) >= n_states_transition:
                hidden_state_sequence = predict(
                    transition_model, transition_data_train, algorithm=algo_predict
                )  # predict hidden state labels
                labels_train[start_end[0] : start_end[1] + 1] = hidden_state_sequence

        # append cleaned sequences to train_sequence
        labels_train_sequence.append(labels_train)

    """Now that we have a fully labeled dataset, we use our already fitted distributions as input for the new model"""

    # extract fitted distributions from both separate trained models
    distributions = get_model_distributions(transition_model) + get_model_distributions(stride_model)

    if init_method == "fully-connected":
        trans_mat, start_probs, end_probs = create_transition_matrix_fully_connected(n_states)

        model_untrained = pg.HiddenMarkovModel.from_matrix(
            transition_probabilities=copy.deepcopy(trans_mat),
            distributions=copy.deepcopy(distributions),
            starts=start_probs,
            ends=None,
            state_names=None,
            verbose=VERBOSE_MODEL,
        )

    if init_method == "labels":

        """Combine already trained transition matrices"""
        # zero pad "stride" transition matrix to the left
        trans_mat_stride = stride_model.dense_transition_matrix()[:-2, :-2]
        transmat_stride = np.pad(
            trans_mat_stride, [(n_states_transition, 0), (n_states_transition, 0)], mode="constant"
        )

        # zero-pad "transition" transition matrix to the right
        trans_mat_transition = transition_model.dense_transition_matrix()[:-2, :-2]
        transmat_trans = np.pad(trans_mat_transition, [(0, n_states_stride), (0, n_states_stride)], mode="constant")

        # after correct zero padding we can combine both transition matrices just by "adding" them together!
        trans_mat = transmat_trans + transmat_stride

        """Find missing transitions from labels"""
        [transitions, starts, ends] = extract_transitions_starts_stops_from_hidden_state_sequence(labels_train_sequence)

        start_probs = np.zeros(n_states)
        start_probs[starts] = 1.0
        end_probs = np.zeros(n_states)
        end_probs[ends] = 1.0

        model_untrained = pg.HiddenMarkovModel.from_matrix(
            transition_probabilities=copy.deepcopy(trans_mat),
            distributions=copy.deepcopy(distributions),
            starts=start_probs,
            ends=None,
            state_names=None,
            verbose=VERBOSE_MODEL,
        )

        """Add missing transitions which will "connect" transition-hmm and stride-hmm"""
        for trans in transitions:
            # if edge already exists, skip
            if not model_untrained.dense_transition_matrix()[trans[0], trans[1]]:
                add_transition(model_untrained, ["s%d" % (trans[0]), "s%d" % (trans[1])], 0.1)

    # pomegranate seems to have a strange sorting bug where state names >= 10 (e.g. s10 get sorted in a bad order like s0, s1, s10, s2 usw..)
    model_untrained = fix_model_names(model_untrained)
    model_untrained.bake()

    # make sure we do not change our distributions anymore!
    model_untrained.freeze_distributions()

    model_trained = copy.deepcopy(model_untrained)

    # convert labels to state-names
    labels_train_sequence_str = labels_to_strings(labels_train_sequence)

    _, history = model_trained.fit(
        sequences=data_train_sequence.copy(),
        labels=labels_train_sequence_str.copy(),
        algorithm=algo_train,
        stop_threshold=stop_threshold,
        max_iterations=max_iterations,
        return_history=True,
        verbose=VERBOSE_MODEL,
        n_jobs=N_JOBS,
    )

    model_trained.name = "HMM-Trained"

    return model_trained, history


def train_hhmm(data_train_sequence, borders_train_sequence, settings):
    check_required_keys_in_settings_dict(
        settings, ["hmm_settings_stride", "hmm_settings_transition", "hmm_settings_combined"]
    )

    """ -------------------- Train STRIDE model -------------------- """
    stride_model_trained = train_stride_hmm(
        data_train_sequence, borders_train_sequence, settings["hmm_settings_stride"]
    )

    """ -------------------- Train TRANSITION model -------------------- """
    transition_model_trained = train_transition_hmm(
        data_train_sequence, borders_train_sequence, settings["hmm_settings_transition"]
    )

    """ -------------------- Train COMBINED model -------------------- """
    model_trained, _ = build_combined_transition_stride_model(
        data_train_sequence,
        borders_train_sequence,
        transition_model_trained,
        stride_model_trained,
        settings["hmm_settings_combined"],
    )

    return [model_trained, stride_model_trained, transition_model_trained]


class HiddenMarkovModel(_BaseSerializable):
    """Wrap all required information to train a new HMM.

    Parameters
    ----------
    model_file_name
        Path to a valid pre-trained and serialized model-json
    sampling_rate_hz_model
        The sampling rate of the data the model was trained with
    low_pass_cutoff_hz
        Cutoff frequency of low-pass filter for preprocessing
    low_pass_order
        Low-pass filter order
    axis
        List of sensor axis which will be used as model input
    features
        List of features which will be used as model input
    window_size_samples
        window size of moving centered window for feature extraction
    standardization
        Flag for feature standardization /  z-score normalization

    See Also
    --------
    TBD

    """

    sampling_rate_hz_model: Optional[float]
    low_pass_cutoff_hz: Optional[float]
    low_pass_order: Optional[int]
    axis: Optional[List[str]]
    features: Optional[List[str]]
    window_size_samples: Optional[int]
    standardization: Optional[bool]
    n_states_stride: Optional[int]
    n_states_transition: Optional[int]

    _model_combined: Optional[pgHMM]
    _model_stride: Optional[pgHMM]
    _model_transition: Optional[pgHMM]

    def __init__(
        self,
        sampling_rate_hz_model: Optional[float] = None,
        low_pass_cutoff_hz: Optional[float] = None,
        low_pass_order: Optional[int] = None,
        axis: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        window_size_samples: Optional[int] = None,
        standardization: Optional[bool] = None,
        n_states_stride: Optional[int] = None,
        n_states_transition: Optional[int] = None,
    ):
        self.sampling_rate_hz_model = sampling_rate_hz_model
        self.low_pass_cutoff_hz = low_pass_cutoff_hz
        self.low_pass_order = low_pass_order
        self.axis = axis
        self.features = features
        self.window_size_samples = window_size_samples
        self.standardization = standardization
        self.n_states_stride: n_states_stride
        self.n_states_transition: n_states_transition

    def predict_hidden_states(self, feature_data, algorithm="viterbi"):
        """Perform prediction based on given data and given model."""
        feature_data = np.ascontiguousarray(feature_data.to_numpy())

        # need to check if memory layout of given data is
        # see related pomegranate issue: https://github.com/jmschrei/pomegranate/issues/717
        if not np.array(feature_data).flags["C_CONTIGUOUS"]:
            raise ValueError("Memory Layout of given input data is not contiguois! Consider using ")

        labels_predicted = np.asarray(self._model_combined.predict(feature_data, algorithm=algorithm))
        # pomegranate always adds an additional label for the start- and end-state, which can be ignored here!
        return np.asarray(labels_predicted[1:-1])

    def train_sub_model(
        self,
        dataset_list,
        n_states,
        n_gmm_components,
        architecture,
        algo_train="baum-welch",
        stop_threshold=1e-9,
        max_iterations=10,
        random_seed=None,
    ):
        return 0

    def train(self, dataset_list, stride_list_list):
        return 0

    @property
    def stride_states_(self):
        """Get list of possible stride state keys."""
        return np.arange(self.n_states_transition, self.n_states_stride + self.n_states_transition)

    @property
    def transition_states_(self):
        """Get list of possible transition state keys."""
        return np.arange(self.n_states_transition)


class HiddenMarkovModelPreTrained(_BaseSerializable):
    """Wrap all required information about a pre-trained HMM.

    Parameters
    ----------
    model_file_name
        Path to a valid pre-trained and serialized model-json
    sampling_rate_hz_model
        The sampling rate of the data the model was trained with
    low_pass_cutoff_hz
        Cutoff frequency of low-pass filter for preprocessing
    low_pass_order
        Low-pass filter order
    axis
        List of sensor axis which will be used as model input
    features
        List of features which will be used as model input
    window_size_samples
        window size of moving centered window for feature extraction
    standardization
        Flag for feature standardization /  z-score normalization

    See Also
    --------
    TBD

    """

    model_file_name: Optional[str]
    sampling_rate_hz_model: Optional[float]
    low_pass_cutoff_hz: Optional[float]
    low_pass_order: Optional[int]
    axis: Optional[List[str]]
    features: Optional[List[str]]
    window_size_samples: Optional[int]
    standardization: Optional[bool]

    _model_combined: Optional[pgHMM]
    _model_stride: Optional[pgHMM]
    _model_transition: Optional[pgHMM]
    _n_states_stride: Optional[int]
    _n_states_transition: Optional[int]

    def __init__(
        self,
        model_file_name: Optional[str] = None,
        sampling_rate_hz_model: Optional[float] = None,
        low_pass_cutoff_hz: Optional[float] = None,
        low_pass_order: Optional[int] = None,
        axis: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        window_size_samples: Optional[int] = None,
        standardization: Optional[bool] = None,
    ):
        self.model_file_name = model_file_name
        self.sampling_rate_hz_model = sampling_rate_hz_model
        self.low_pass_cutoff_hz = low_pass_cutoff_hz
        self.low_pass_order = low_pass_order
        self.axis = axis
        self.features = features
        self.window_size_samples = window_size_samples
        self.standardization = standardization

        # try to load models
        with open_text("gaitmap.stride_segmentation.hmm_models", self.model_file_name) as test_data:
            with open(test_data.name) as f:
                models_dict = json.load(f)

        if (
            "combined_model" not in models_dict.keys()
            or "stride_model" not in models_dict.keys()
            or "transition_model" not in models_dict.keys()
        ):
            raise ValueError(
                "Invalid model-json! Keys within the model-json required are: 'combined_model', 'stride_model' and "
                "'transition_model'"
            )

        self._model_stride = pgHMM.from_json(models_dict["stride_model"])
        self._model_transition = pgHMM.from_json(models_dict["transition_model"])
        self._model_combined = pgHMM.from_json(models_dict["combined_model"])

        # we need to subtract the silent start and end state form the state count!
        self._n_states_stride = self._model_stride.state_count() - 2
        self._n_states_transition = self._model_transition.state_count() - 2

    def predict_hidden_states(self, feature_data, algorithm="viterbi"):
        """Perform prediction based on given data and given model."""
        feature_data = np.ascontiguousarray(feature_data.to_numpy())

        # need to check if memory layout of given data is
        # see related pomegranate issue: https://github.com/jmschrei/pomegranate/issues/717
        if not np.array(feature_data).flags["C_CONTIGUOUS"]:
            raise ValueError("Memory Layout of given input data is not contiguois! Consider using ")

        labels_predicted = np.asarray(self._model_combined.predict(feature_data, algorithm=algorithm))
        # pomegranate always adds an additional label for the start- and end-state, which can be ignored here!
        return np.asarray(labels_predicted[1:-1])

    @property
    def stride_states_(self):
        """Get list of possible stride state keys."""
        return np.arange(self._n_states_transition, self._n_states_stride + self._n_states_transition)

    @property
    def transition_states_(self):
        """Get list of possible transition state keys."""
        return np.arange(self._n_states_transition)


class HiddenMarkovModelStairs(HiddenMarkovModelPreTrained):
    """Hidden Markov Model trained for stride segmentation including stair strides.

    Notes
    -----
    This is a pre-trained model aiming to segment also "stair strides" next to "normal strides"

    See Also
    --------
    TBD

    """

    model_file_name = "hmm_stairs.json"
    # preprocessing settings
    sampling_rate_hz_model = 51.2
    low_pass_cutoff_hz = 10.0
    low_pass_order = 4
    # feature settings
    axis = ["gyr_ml"]
    features = ["raw", "gradient"]
    window_size_samples = 11
    standardization = True

    def __init__(self):
        super().__init__(
            model_file_name=self.model_file_name,
            sampling_rate_hz_model=self.sampling_rate_hz_model,
            low_pass_cutoff_hz=self.low_pass_cutoff_hz,
            low_pass_order=self.low_pass_order,
            axis=self.axis,
            features=self.features,
            window_size_samples=self.window_size_samples,
            standardization=self.standardization,
        )


class HiddenMarkovModelStairs(HiddenMarkovModel):
    """Hidden Markov Model trained for stride segmentation including stair strides.

    Notes
    -----
    This is a pre-trained model aiming to segment also "stair strides" next to "normal strides"

    See Also
    --------
    TBD

    """

    model_file_name = "hmm_stairs.json"
    # preprocessing settings
    sampling_rate_hz_model = 51.2
    low_pass_cutoff_hz = 10.0
    low_pass_order = 4
    # feature settings
    axis = ["gyr_ml"]
    features = ["raw", "gradient"]
    window_size_samples = 11
    standardization = True

    def __init__(self):
        super().__init__(
            model_file_name=self.model_file_name,
            sampling_rate_hz_model=self.sampling_rate_hz_model,
            low_pass_cutoff_hz=self.low_pass_cutoff_hz,
            low_pass_order=self.low_pass_order,
            axis=self.axis,
            features=self.features,
            window_size_samples=self.window_size_samples,
            standardization=self.standardization,
        )
