"""Segmentation model base classes and helper."""
import copy
import json
from importlib.resources import open_text
from typing import Optional

import numpy as np
import pomegranate as pg
from pomegranate import HiddenMarkovModel as pgHMM

from gaitmap.base import _BaseSerializable
from gaitmap_mad.stride_segmentation._hmm.hmm_feature_transform import FeatureTransformHMM
from gaitmap_mad.stride_segmentation._hmm.simple_model import SimpleHMM
from gaitmap_mad.stride_segmentation._hmm.utils import (
    add_transition,
    create_transition_matrix_fully_connected,
    extract_transitions_starts_stops_from_hidden_state_sequence,
    fix_model_names,
    get_model_distributions,
    get_train_data_sequences_strides,
    get_train_data_sequences_transitions,
    labels_to_strings,
)
from gaitmap.utils.array_handling import bool_array_to_start_end_array, start_end_array_to_bool_array

N_JOBS = 1


def create_fully_labeled_gait_sequences(
    data_train_sequence, stride_list_sequence, transition_model, stride_model, algo_predict
):
    """Create fully labeled gait sequence.

    To find the "actual" hidden-state labels for "labeled-training" with the given training data set, we will again
    split everything into strides and transitions based on our initial stride borders and then predict the labels with
    the respective already learned models.

    To rephrase it again: We want to create a fully labeled dataset with already optimal hidden-state labels, but as
    these lables are hidden, we need to predict them with our already trained models...
    """
    labels_train_sequence = []

    for data, stride_list in zip(data_train_sequence, stride_list_sequence):
        labels_train = np.zeros(len(data))

        # predict hidden-state sequence for each transition using "transition model"
        transition_mask = np.invert(
            start_end_array_to_bool_array(stride_list[["start", "end"]].to_numpy(), pad_to_length=len(data) - 1)
        )
        transition_start_end_list = bool_array_to_start_end_array(transition_mask)

        # for each transition, get data and create some naive labels for initialization
        for start, end in transition_start_end_list:
            transition_data_train = data[start:end]
            labels_train[start:end] = transition_model.predict_hidden_state_sequence(
                transition_data_train, algorithm=algo_predict
            )

        # predict hidden-state sequence for each stride using "stride model"
        for start, end in stride_list[["start", "end"]].to_numpy():
            stride_data_train = data[start:end]
            labels_train[start:end] = (
                stride_model.predict_hidden_state_sequence(stride_data_train, algorithm=algo_predict)
                + transition_model.n_states
            )

        # append cleaned sequences to train_sequence
        labels_train_sequence.append(labels_train)

    return labels_train_sequence


class SimpleSegmentationHMM(_BaseSerializable):
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
    window_size_s
        window size of moving centered window for feature extraction
    standardization
        Flag for feature standardization /  z-score normalization

    See Also
    --------
    TBD

    """

    stride_model: Optional[SimpleHMM]
    transition_model: Optional[SimpleHMM]
    feature_transform: Optional[FeatureTransformHMM]
    algo_predict: Optional[str]
    algo_train: Optional[str]
    stop_threshold: Optional[float]
    max_iterations: Optional[int]
    initialization: Optional[str]
    name: Optional[str]
    model: Optional[pgHMM]

    # TODO: What are those?
    n_states: Optional[int]
    history: Optional[pg.callbacks.History]

    def __init__(
        self,
        stride_model: Optional[SimpleHMM] = None,
        transition_model: Optional[SimpleHMM] = None,
        feature_transform: Optional[FeatureTransformHMM] = None,
        algo_predict: Optional[str] = None,
        algo_train: Optional[str] = None,
        stop_threshold: Optional[float] = None,
        max_iterations: Optional[int] = None,
        initialization: Optional[str] = None,
        verbose: bool = True,
        name: Optional[str] = None,
        model: Optional[pgHMM] = None,
    ):
        self.stride_model = stride_model
        self.transition_model = transition_model
        self.feature_transform = feature_transform
        self.algo_predict = algo_predict
        self.algo_train = algo_train
        self.stop_threshold = stop_threshold
        self.max_iterations = max_iterations
        self.initialization = initialization
        self.verbose = verbose
        self.name = name
        self.model = model

    @property
    def stride_states_(self) -> dict:
        """Return stride states."""
        return np.arange(self.stride_model.n_states) + self.transition_model.n_states

    @property
    def transition_states_(self) -> dict:
        """Return transition states."""
        return np.arange(self.transition_model.n_states)

    def predict_hidden_state_sequence(self, feature_data):
        """Perform prediction based on given data and given model."""
        if self.model is None:
            raise ValueError(
                "No trained model for prediction available! You must either provide a pre-trained model "
                "during class initialization or call the train method with appropriate training data to "
                "generate a new trained model."
            )

        feature_data = np.ascontiguousarray(feature_data.to_numpy())

        # need to check if memory layout of given data is
        # see related pomegranate issue: https://github.com/jmschrei/pomegranate/issues/717
        if not np.array(feature_data).data.c_contiguous:
            raise ValueError("Memory Layout of given input data is not contiguois! Consider using ")

        labels_predicted = np.asarray(self.model.predict(feature_data.copy(), algorithm=self.algo_predict))
        # pomegranate always adds an additional label for the start- and end-state, which can be ignored here!
        return np.asarray(labels_predicted[1:-1])

    def transform(self, data_sequence, stride_list_sequence, sampling_frequency_hz):
        """Perform feature transformation."""
        if not isinstance(data_sequence, list):
            raise ValueError("Input into transform must be a list of valid gaitmapt sensordata objects!")

        data_sequence_feature_space = [
            self.feature_transform.transform(dataset, sampling_rate_hz=sampling_frequency_hz) for dataset in data_sequence
        ]

        stride_list_feature_space = None
        if stride_list_sequence:
            downsample_factor = int(
                np.round(sampling_frequency_hz / self.feature_transform.sampling_rate_feature_space_hz)
            )
            stride_list_feature_space = [
                (stride_list[["start", "end"]] / downsample_factor).astype(int) for stride_list in stride_list_sequence
            ]
        return [data_sequence_feature_space, stride_list_feature_space]

    def train(self, data_sequence, stride_list_sequence, sampling_frequency_hz):
        """Train HMM."""
        # perform feature transformation
        data_sequence_feature_space, stride_list_feature_space = self.transform(
            data_sequence, stride_list_sequence, sampling_frequency_hz
        )

        if self.initialization not in ["labels", "fully-connected"]:
            raise ValueError('Invalid value for initialization! Must be one of "labels" or "fully-connected"')

        # train sub stride model
        strides_sequence, init_stride_state_labels = get_train_data_sequences_strides(
            data_sequence_feature_space, stride_list_feature_space, self.stride_model.n_states
        )

        stride_model_trained = self.stride_model.build_model(strides_sequence, init_stride_state_labels)

        # train sub transition model
        transition_sequence, init_trans_state_labels = get_train_data_sequences_transitions(
            data_sequence_feature_space, stride_list_feature_space, self.transition_model.n_states
        )

        transition_model_trained = self.transition_model.build_model(transition_sequence, init_trans_state_labels)

        # For model combination actually only the transition probabilities will be updated, while keeping the already
        # learned distributions for all states. This can be achieved by "labeled" training, where basically just the
        # number of transitions will be counted.

        # some initialization stuff...
        self.n_states = transition_model_trained.n_states + stride_model_trained.n_states
        n_states_transition = transition_model_trained.n_states
        n_states_stride = stride_model_trained.n_states

        # extract fitted distributions from both separate trained models
        distributions = get_model_distributions(transition_model_trained.model) + get_model_distributions(
            stride_model_trained.model
        )

        # predict hidden state labels for complete walking bouts
        labels_train_sequence = create_fully_labeled_gait_sequences(
            data_sequence_feature_space,
            stride_list_feature_space,
            transition_model_trained,
            stride_model_trained,
            self.algo_predict,
        )

        # Now that we have a fully labeled dataset, we use our already fitted distributions as input for the new model
        if self.initialization == "fully-connected":
            trans_mat, start_probs, end_probs = create_transition_matrix_fully_connected(self.n_states)

            model_untrained = pg.HiddenMarkovModel.from_matrix(
                transition_probabilities=copy.deepcopy(trans_mat),
                distributions=copy.deepcopy(distributions),
                starts=start_probs,
                ends=None,
                state_names=None,
                verbose=self.verbose,
            )

        if self.initialization == "labels":
            # combine already trained transition matrices -> zero pad "stride" transition matrix to the left
            trans_mat_stride = stride_model_trained.model.dense_transition_matrix()[:-2, :-2]
            transmat_stride = np.pad(
                trans_mat_stride, [(n_states_transition, 0), (n_states_transition, 0)], mode="constant"
            )

            # zero-pad "transition" transition matrix to the right
            trans_mat_transition = transition_model_trained.model.dense_transition_matrix()[:-2, :-2]
            transmat_trans = np.pad(trans_mat_transition, [(0, n_states_stride), (0, n_states_stride)], mode="constant")

            # after correct zero padding we can combine both transition matrices just by "adding" them together!
            trans_mat = transmat_trans + transmat_stride

            # find missing transitions from labels
            [transitions, starts, ends] = extract_transitions_starts_stops_from_hidden_state_sequence(
                labels_train_sequence
            )

            start_probs = np.zeros(self.n_states)
            start_probs[starts] = 1.0
            end_probs = np.zeros(self.n_states)
            end_probs[ends] = 1.0

            model_untrained = pg.HiddenMarkovModel.from_matrix(
                transition_probabilities=copy.deepcopy(trans_mat),
                distributions=copy.deepcopy(distributions),
                starts=start_probs,
                ends=None,
                state_names=None,
                verbose=self.verbose,
            )

            # Add missing transitions which will "connect" transition-hmm and stride-hmm
            for trans in transitions:
                # if edge already exists, skip
                if not model_untrained.dense_transition_matrix()[trans[0], trans[1]]:
                    add_transition(model_untrained, ["s%d" % (trans[0]), "s%d" % (trans[1])], 0.1)

        # pomegranate seems to have a strange sorting bug where state names >= 10 (e.g. s10 get sorted in a bad order
        # like s0, s1, s10, s2 usw..)
        model_untrained = fix_model_names(model_untrained)
        model_untrained.bake()

        # make sure we do not change our distributions anymore!
        model_untrained.freeze_distributions()

        model_trained = copy.deepcopy(model_untrained)

        # convert labels to state-names
        labels_train_sequence_str = labels_to_strings(labels_train_sequence)

        # make sure data is in an pomegranate compatible format!
        data_train_sequence = [
            np.ascontiguousarray(copy.deepcopy(feature_data.to_numpy())) for feature_data in data_sequence_feature_space
        ]

        _, history = model_trained.fit(
            sequences=data_train_sequence,
            labels=labels_train_sequence_str.copy(),
            algorithm=self.algo_train,
            stop_threshold=self.stop_threshold,
            max_iterations=self.max_iterations,
            return_history=True,
            verbose=self.verbose,
            n_jobs=N_JOBS,
        )

        model_trained.name = self.name

        self.history = history
        self.model = model_trained

        return self

    def export_json(self, path):
        """Export HMM to json file."""
        with open(path, "w") as f:
            json.dump(self.to_json(), f)


class PreTrainedSegmentationHMM(SimpleSegmentationHMM):
    """Load a pre-trained stride segmentation HMM."""

    def __init__(self):
        super().__init__()

    def __new__(cls, model_file_name="fallriskpd_at_lab_model.json"):
        # try to load models
        with open_text("gaitmap_mad.stride_segmentation._hmm._pre_trained_models", model_file_name) as test_data:
            with open(test_data.name) as f:
                model_json = f.read()
        return SimpleSegmentationHMM.from_json(model_json)
