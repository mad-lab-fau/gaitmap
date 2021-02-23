"""Dtw template base classes and helper."""
from importlib.resources import open_text
from typing import Optional, List
import json
from pomegranate import HiddenMarkovModel as pgHMM
import pomegranate as pg
import numpy as np
from gaitmap.base import _BaseSerializable
from gaitmap.future.hmm.simple_model import (
    build_and_train_sub_hmm,
)
from gaitmap.utils.array_handling import bool_array_to_start_end_array, start_end_array_to_bool_array

import copy

VERBOSE_MODEL = True
VERBOSE_DISTRIBUTIONS = False
DEBUG_PLOTS = False
N_JOBS = 1


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


def create_fully_labeled_gait_sequences(
    data_train_sequence, stride_list_sequence, transition_model, stride_model, algo_predict
):
    """To find the "actual" hidden-state labels for "labeled-training" with the given training data set, we will again
    split everything into strides and transitions based on our initial stride borders and then predict the labels with
    the respective already learned models.

    To rephrase it again: We want to create a fully labeled dataset with already optimal hidden-state labels, but as these
    lables are hidden, we need to predict them with our already trained models...
    """

    n_states_transition = len(transition_model.states) - 2  # subtract silent start- and end-state

    labels_train_sequence = []

    for data, stride_list in zip(data_train_sequence, stride_list_sequence):
        labels_train = np.zeros(len(data))

        # predict hidden-state sequence for each stride using "stride model"
        for start, end in stride_list[["start", "end"]].to_numpy():
            stride_data_train = data[start:end]
            labels_train[start:end] = (
                predict(stride_model, stride_data_train, algorithm=algo_predict) + n_states_transition
            )

        # predict hidden-state sequence for each transition using "transition model"
        transition_mask = np.invert(
            start_end_array_to_bool_array(stride_list[["start", "end"]].to_numpy(), pad_to_length=len(data) - 1)
        )
        transition_start_end_list = bool_array_to_start_end_array(transition_mask)

        # for each transition, get data and create some naive labels for initialization
        for start, end in transition_start_end_list:
            transition_data_train = data[start : end + 1]
            labels_train[start : end + 1] = predict(transition_model, transition_data_train, algorithm=algo_predict)

        # append cleaned sequences to train_sequence
        labels_train_sequence.append(labels_train)

    return labels_train_sequence


def build_combined_transition_stride_model(
    data_train_sequence,
    stride_list_sequence,
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

    # extract fitted distributions from both separate trained models
    distributions = get_model_distributions(transition_model) + get_model_distributions(stride_model)

    # predict hidden state labels for complete walking bouts
    labels_train_sequence = create_fully_labeled_gait_sequences(
        data_train_sequence, stride_list_sequence, transition_model, stride_model, algo_predict
    )

    """Now that we have a fully labeled dataset, we use our already fitted distributions as input for the new model"""
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


class SimpleHMM(_BaseSerializable):
    """Wrap all required information to train a new HMM.

    Parameters
    ----------
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

    n_states: Optional[int]
    n_gmm_components: Optional[int]
    algo_train: Optional[str]
    algo_predict: Optional[str]
    stop_threshold: Optional[float]
    max_iterations: Optional[int]
    random_seed: Optional[float]
    architecture: Optional[str]
    name: Optional[str]

    def __init__(
        self,
        n_states: Optional[int] = None,
        n_gmm_components: Optional[int] = None,
        algo_train: Optional[str] = None,
        algo_predict: Optional[str] = None,
        stop_threshold: Optional[float] = None,
        max_iterations: Optional[int] = None,
        random_seed: Optional[float] = None,
        architecture: Optional[str] = None,
        name: Optional[str] = None,
        model: Optional[pgHMM] = None,
    ):
        self.n_states = (n_states,)
        self.n_gmm_components = (n_gmm_components,)
        self.algo_train = (algo_train,)
        self.algo_predict = (algo_predict,)
        self.stop_threshold = (stop_threshold,)
        self.max_iterations = (max_iterations,)
        self.random_seed = (random_seed,)
        self.architecture = (architecture,)
        self.name = (name,)
        self.model = None

    def predict(self, feature_data, algorithm="viterbi"):
        """Perform prediction based on given data and given model."""
        feature_data = np.ascontiguousarray(feature_data.to_numpy())

        # need to check if memory layout of given data is
        # see related pomegranate issue: https://github.com/jmschrei/pomegranate/issues/717
        if not np.array(feature_data).flags["C_CONTIGUOUS"]:
            raise ValueError("Memory Layout of given input data is not contiguois! Consider using ")

        labels_predicted = np.asarray(self._model_combined.predict(feature_data, algorithm=algorithm))
        # pomegranate always adds an additional label for the start- and end-state, which can be ignored here!
        return np.asarray(labels_predicted[1:-1])

    def build_model(self, data_sequence, labels_sequence):
        return self


class HiddenMarkovModel(_BaseSerializable):
    """Wrap all required information to train a new HMM.

    Parameters
    ----------
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
