"""Segmentation _model base classes and helper."""
import copy
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pomegranate as pg
from pomegranate import HiddenMarkovModel as pgHMM
from pomegranate.hmm import History
from tpcp import OptiPara, cf, make_optimize_safe
from typing_extensions import Self

from gaitmap.base import _BaseSerializable
from gaitmap.utils.datatype_helper import SingleSensorData, SingleSensorStrideList
from gaitmap_mad.stride_segmentation.hmm._hmm_feature_transform import (
    BaseHmmFeatureTransformer,
    RothHmmFeatureTransformer,
)
from gaitmap_mad.stride_segmentation.hmm._simple_model import SimpleHmm
from gaitmap_mad.stride_segmentation.hmm._utils import (
    ShortenedHMMPrint,
    _clone_model,
    _DataToShortError,
    _HackyClonableHMMFix,
    add_transition,
    convert_stride_list_to_transition_list,
    create_transition_matrix_fully_connected,
    extract_transitions_starts_stops_from_hidden_state_sequence,
    fix_model_names,
    get_model_distributions,
    get_train_data_sequences_strides,
    get_train_data_sequences_transitions,
    labels_to_strings,
    predict,
)


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
        transition_start_end_list = convert_stride_list_to_transition_list(stride_list, data.shape[0])

        # for each transition, get data and create some naive labels for initialization
        for start, end in transition_start_end_list[["start", "end"]].to_numpy():
            transition_data_train = data[start:end]
            try:
                labels_train[start:end] = transition_model.predict_hidden_state_sequence(
                    transition_data_train, algorithm=algo_predict
                )
            except _DataToShortError:
                # This happens if a transition is too short to be predicted by the transition model
                continue

        # predict hidden-state sequence for each stride using "stride model"
        for start, end in stride_list[["start", "end"]].to_numpy():
            stride_data_train = data[start:end]
            try:
                labels_train[start:end] = (
                    stride_model.predict_hidden_state_sequence(stride_data_train, algorithm=algo_predict)
                    + transition_model.n_states
                )
            except _DataToShortError:
                # This happens if a stride is too short to be predicted by the stride model
                continue

        # append cleaned sequences to train_sequence
        labels_train_sequence.append(labels_train)

    return labels_train_sequence


class BaseSegmentationHmm(_BaseSerializable):
    """Base class for HMM segmentation models.

    In case you want to propose your own HMM architecture that is not covered by the options provided in
    :class:`~gaitmap.stride_segmentation.hmm.RothSegmentationHmm`, you can inherit from this class and implement all
    abstract methods.
    """

    _action_methods = ("predict",)

    hidden_state_sequence_: np.ndarray

    data: pd.DataFrame
    sampling_rate_hz: float

    @property
    def stride_states(self) -> List[int]:
        """Get the indexes of the states in the hidden state sequence corresponding to strides."""
        raise NotImplementedError

    def predict(self, data: SingleSensorData, sampling_rate_hz: float) -> Self:
        """Perform prediction based on given data and given model.

        Parameters
        ----------
        data
            The data to predict the hidden state sequence for.
        sampling_rate_hz
            The sampling rate of the data.

        Returns
        -------
        self
            The instance with the result objects attached.

        """
        raise NotImplementedError

    def self_optimize(
        self,
        data_sequence: Sequence[SingleSensorData],
        stride_list_sequence: Sequence[SingleSensorStrideList],
        sampling_rate_hz: float,
    ) -> Self:
        """Create and train the HMM model based on the given data and labels.

        Parameters
        ----------
        data_sequence
            Sequence of gaitmap sensordata objects.
        stride_list_sequence
            Sequence of gaitmap stride lists.
            The number of stride lists must match the number of sensordata objects (i.e. they must belong together).
        sampling_rate_hz
            Sampling frequency of the data.

        Returns
        -------
        self
            The trained model instance.

        """
        raise NotImplementedError

    def self_optimize_with_info(
        self,
        data_sequence: Sequence[SingleSensorData],
        stride_list_sequence: Sequence[SingleSensorStrideList],
        sampling_rate_hz: float,
    ) -> Tuple[Self, Any]:
        """Create and train the HMM model based on the given data and labels.

        This is identical to `self_optimize`, but can return additional information about the training process.

        Parameters
        ----------
        data_sequence
            Sequence of gaitmap sensordata objects.
        stride_list_sequence
            Sequence of gaitmap stride lists.
            The number of stride lists must match the number of sensordata objects (i.e. they must belong together).
        sampling_rate_hz
            Sampling frequency of the data.

        Returns
        -------
        self
            The trained model instance.
        training_info
            An arbitrary object containing training information.

        """
        raise NotImplementedError


class RothSegmentationHmm(BaseSegmentationHmm, _HackyClonableHMMFix, ShortenedHMMPrint):
    """A hierarchical HMM model for stride segmentation proposed by Roth et al. (2019).

    This model differentiates between strides and transitions.
    Both data sections are modeled by individual HMMs and are trained seperately.
    A final model is created by combining the transition matrizes of the two models and allowing for transitions between
    these higher level states at the start or end of a stride.

    Note, that we are also store the trained stride and transition model during the optimization step.
    These are not required for prediction, as all there information is also contained in the fused model.
    However, inspecting the trained models might provide further inside into possible training issues.
    As the models are generally small, this should not impact RAM (or disk usage during export) in a relevant way.

    Parameters
    ----------
    stride_model
        The (untrained) hmm representing the strides in the data.
        This will be updated during the optimization step (see notes).
    transition_model
        The (untrained) hmm representing the transitions in the data.
        This will be updated during the optimization step (see notes).
    feature_transform
        An instance of a :class:`~gaitmap.stride_segmentation.hmm.FeatureTransformHMM` that can transform the data
        (and the labeled stride list) into the feature space required by the HMM.
        If you want to use custome feature extraction,
    algo_train
        The algorithm to use for training the HMM.
    algo_predict
        The algorithm to use for prediction with the HMM.
    stop_threshold
        The loss threshold to stop the optimization.
        Note, that is the threshold for the "combined" training of the final model.
        This is less important, as we recommend to train the combined model only for a single iteration anyway.
        If you want to adjust the stop threshold for the individual models, you can do so by adjusting the respective
        parameters of the stride and transition model (`stride_model.stop_threshold` and
        `transition_model.stop_threshold`).
    max_iterations
        The maximum number of iterations to perform during the optimization.
        Note, that this is the value for the "combined" training of the final model.
        We recommend keeping this value at 1, as the combined model training only adjusts the transition matrix.
        If you want to adjust the max iterations for the individual models, you can do so by adjusting the respective
        parameters of the stride and transition model (`stride_model.max_iterations` and
        `transition_model.max_iterations`).
    initialization
        The initialization method to use for the HMM during optimization.
        `fully-connected` assumes that all states are reachable from any other state with the same probability.
        `labels` will derive the allowed transitions and probabilities from the given labels.
        In both cases, distributions will be derived from the data during optimization.
        We recommend using `labels` here, as this is kind of the default mode we expect these segmentation models to be
        used.
        If you select `fully-connected`, you might want to increase the `max_iterations` parameter to allow the model to
        actually be trained.
    verbose
        If True, print additional information during optimization.
    n_jobs
        The number of parallel jobs to use during optimization.
        If set to -1, all available cores will be used.
    name
        The name of the final pomegranate model.
    model
        The actual pomegranate HMM model.
        This can be set to `None` initially.
        A model will then be created during the optimization step.
        If you want to use a pre-trained model, you can set this parameter to the respective model.
        However, we recommend to ideally export this entire class instead of just the model to make sure that things
        like the feature transform are also exported/stored.
    data_columns
        The expected columns of the input data in feature space.
        This will be automatically set based on the feature transform output during the optimization step.
        This does not affect the output, but is used as a sanity check to ensure that valid input data is provided
        and that the column order is correct.

    Attributes
    ----------
    feature_space_data_
        The data in feature space.
        This is only here for debugging purposes.
    hidden_state_sequence_
        The hidden state sequence predicted by the model with the same sampling rate as the input data.
    hidden_state_sequence_feature_space_
        The predicted hidden-state sequence in feature space.


    Other Parameters
    ----------------
    data
        The data passed to the `segment` method.
    sampling_rate_hz
        The sampling rate of the data

    """

    stride_model: SimpleHmm
    stride_model__model: OptiPara
    stride_model__data_columns: OptiPara
    transition_model: SimpleHmm
    transition_model__model: OptiPara
    transition_model__data_columns: OptiPara
    feature_transform: BaseHmmFeatureTransformer
    algo_predict: Literal["viterbi", "baum-welch"]
    algo_train: Literal["viterbi", "baum-welch"]
    stop_threshold: float
    max_iterations: int
    initialization: Literal["labels", "fully-connected"]
    verbose: bool
    n_jobs: int
    name: Optional[str]
    model: OptiPara[Optional[pgHMM]]
    data_columns: OptiPara[Optional[Tuple[str, ...]]]

    feature_space_data_: pd.DataFrame
    hidden_state_sequence_feature_space_: np.ndarray

    def __init__(
        self,
        stride_model: SimpleHmm = cf(
            SimpleHmm(
                n_states=20,
                n_gmm_components=6,
                algo_train="baum-welch",
                stop_threshold=1e-9,
                max_iterations=10,
                architecture="left-right-strict",
                name="stride_model",
            )
        ),
        transition_model: SimpleHmm = cf(
            SimpleHmm(
                n_states=5,
                n_gmm_components=3,
                algo_train="baum-welch",
                stop_threshold=1e-9,
                max_iterations=10,
                architecture="left-right-loose",
                name="transition_model",
            )
        ),
        feature_transform: RothHmmFeatureTransformer = cf(RothHmmFeatureTransformer()),
        *,
        algo_predict: Literal["viterbi", "map"] = "viterbi",
        algo_train: Literal["viterbi", "baum-welch"] = "baum-welch",
        stop_threshold: float = 1e-9,
        max_iterations: int = 1,
        initialization: Literal["labels", "fully-connected"] = "labels",
        verbose: bool = True,
        n_jobs: int = 1,
        name: str = "segmentation_model",
        model: Optional[pgHMM] = None,
        data_columns: Optional[Tuple[str, ...]] = None,
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
        self.n_jobs = n_jobs
        self.name = name
        self.model = model
        self.data_columns = data_columns

    @property
    def n_states(self) -> int:
        """Return the number of states of the final model."""
        return self.transition_model.n_states + self.stride_model.n_states

    @property
    def stride_states(self) -> List[int]:
        """Return the ids of the stride states."""
        return (np.arange(self.stride_model.n_states) + self.transition_model.n_states).tolist()

    @property
    def transition_states(self) -> List[int]:
        """Return the ids of the transition states."""
        return np.arange(self.transition_model.n_states).tolist()

    def predict(self, data: SingleSensorData, sampling_rate_hz: float) -> Self:
        """Perform prediction based on given data and given model.

        This generates the hidden state sequence and stores it in the `hidden_state_sequence_` attribute.
        Data will first be transformed using the feature transform and then the trained model will be used to predict
        the individual hidden states.

        Parameters
        ----------
        data
            The data to predict the hidden state sequence for.
            Note, that this must have the same columns than the data used during training.
        sampling_rate_hz
            The sampling rate of the data.

        Returns
        -------
        self
            The instance with the result objects attached.

        """
        if self.model is None:
            # We perform this check early to terminate before the potentially costly feature transform
            raise ValueError(
                "No trained model for prediction available! "
                "You must either provide a pre-trained model during class initialization or call the "
                "`self_optimize`/`self_optimize_with_info` method with appropriate training data to generate a new "
                "trained model."
            )

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        feature_data, _ = self._transform([data], None, sampling_rate_hz=sampling_rate_hz)
        feature_data = feature_data[0]
        self.feature_space_data_ = feature_data

        # pomegranate always adds a label for the start- and end-state, which can be ignored here!
        self.hidden_state_sequence_feature_space_ = predict(
            self.model, feature_data, expected_columns=self.data_columns, algorithm=self.algo_predict
        )
        self.hidden_state_sequence_ = self.feature_transform.inverse_transform_state_sequence(
            self.hidden_state_sequence_feature_space_, sampling_rate_hz=sampling_rate_hz
        )
        return self

    def _transform(
        self,
        data_sequence: Sequence[pd.DataFrame],
        stride_list_sequence: Optional[Sequence[pd.DataFrame]],
        sampling_rate_hz: float,
    ):
        """Perform feature transformation."""
        feature_transform = self.feature_transform.clone()

        data_sequence_feature_space = [
            feature_transform.transform(dataset, sampling_rate_hz=sampling_rate_hz).transformed_data_
            for dataset in data_sequence
        ]

        stride_list_feature_space = None
        if stride_list_sequence:
            stride_list_feature_space = [
                feature_transform.transform(
                    roi_list=stride_list, sampling_rate_hz=sampling_rate_hz
                ).transformed_roi_list_
                for stride_list in stride_list_sequence
            ]
        return data_sequence_feature_space, stride_list_feature_space

    def self_optimize(
        self,
        data_sequence: Sequence[SingleSensorData],
        stride_list_sequence: Sequence[SingleSensorStrideList],
        sampling_rate_hz: float,
    ) -> Self:
        """Create and train the HMM model based on the given data and labels.

        This will first apply the feature transformation to the given data and then train the HMM model in three steps:

        1. Train the stride model on the stride data
        2. Train the transition model on the transition data
        3. Assemble the final model by combining the stride and transition model and train it for a couple further
           iterations

        Parameters
        ----------
        data_sequence
            Sequence of gaitmap sensordata objects.
        stride_list_sequence
            Sequence of gaitmap stride lists.
            The number of stride lists must match the number of sensordata objects (i.e. they must belong together).
        sampling_rate_hz
            Sampling frequency of the data.

        Returns
        -------
        self
            The trained model instance.

        """
        return self.self_optimize_with_info(data_sequence, stride_list_sequence, sampling_rate_hz=sampling_rate_hz)[0]

    @make_optimize_safe
    def self_optimize_with_info(
        self,
        data_sequence: Sequence[SingleSensorData],
        stride_list_sequence: Sequence[SingleSensorStrideList],
        sampling_rate_hz: float,
    ) -> Tuple[Self, Dict[Literal["self", "transition_model", "stride_model"], History]]:
        """Create and train the HMM model based on the given data and labels.

        This is identical to `self_optimize`, but returns additional information about the training process.
        The dictionary returned as second parameter contains the training history for each of the three models (
        stride-model, transition-model, and the combined final model "self").

        Parameters
        ----------
        data_sequence
            Sequence of gaitmap sensordata objects.
        stride_list_sequence
            Sequence of gaitmap stride lists.
            The number of stride lists must match the number of sensordata objects (i.e. they must belong together).
        sampling_rate_hz
            Sampling frequency of the data.

        Returns
        -------
        self
            The trained model instance.
        history
            Dictionary containing the training history for each of the three models

        """
        if self.initialization not in ["labels", "fully-connected"]:
            raise ValueError("Invalid value for initialization! Must be one of `labels` or `fully-connected`.")

        # perform feature transformation
        data_sequence_feature_space, stride_list_feature_space = self._transform(
            data_sequence, stride_list_sequence, sampling_rate_hz
        )

        # train sub stride model
        strides_sequence, init_stride_state_labels = get_train_data_sequences_strides(
            data_sequence_feature_space, stride_list_feature_space, self.stride_model.n_states
        )

        stride_model_trained, stride_model_history = self.stride_model.self_optimize_with_info(
            strides_sequence, init_stride_state_labels
        )

        # train sub transition _model
        transition_sequence, init_trans_state_labels = get_train_data_sequences_transitions(
            data_sequence_feature_space, stride_list_feature_space, self.transition_model.n_states
        )

        transition_model_trained, transition_model_history = self.transition_model.self_optimize_with_info(
            transition_sequence, init_trans_state_labels
        )

        # For model combination actually only the transition probabilities will be updated, while keeping the already
        # learned distributions for all states. This can be achieved by "labeled" training, where basically just the
        # number of transitions will be counted.

        # some initialization stuff...
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

            new_model = pg.HiddenMarkovModel.from_matrix(
                transition_probabilities=copy.deepcopy(trans_mat),
                distributions=copy.deepcopy(distributions),
                starts=start_probs,
                ends=None,
                state_names=None,
                verbose=self.verbose,
            )

        elif self.initialization == "labels":
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
            transitions, starts, ends = extract_transitions_starts_stops_from_hidden_state_sequence(
                labels_train_sequence
            )

            start_probs = np.zeros(self.n_states)
            start_probs[starts] = 1.0
            end_probs = np.zeros(self.n_states)
            end_probs[ends] = 1.0

            new_model = pg.HiddenMarkovModel.from_matrix(
                transition_probabilities=copy.deepcopy(trans_mat),
                distributions=copy.deepcopy(distributions),
                starts=start_probs,
                ends=None,
                state_names=None,
                verbose=self.verbose,
            )

            existing_transitions = {(start.name, end.name) for start, end in new_model.graph.edges()}
            missing_transitions = transitions - existing_transitions
            # Add missing transitions which will "connect" transition-hmm and stride-hmm
            # We initialize with a very small probability, so that the model can learn the correct values in the next
            # step.
            # Note: We sort the transitions to enforce consistent order and reproducibility.
            for trans in sorted(missing_transitions):
                add_transition(new_model, trans, 0.1)
        else:
            # Can not be reached, as we perform the check beforehand, but just to be sure and make the linter happy
            raise RuntimeError()
        # pomegranate seems to have a strange sorting bug where state names >= 10 (e.g. s10 get sorted in a bad order
        # like s0, s1, s10, s2 usw..)
        new_model = fix_model_names(new_model)
        new_model.bake()

        # make sure we do not change our distributions anymore!
        new_model.freeze_distributions()

        # We clone the model here, as this changes the order of edges to be sorted somehow...
        new_model = _clone_model(new_model, assert_correct=False)

        # convert labels to state-names
        labels_train_sequence_str = labels_to_strings(labels_train_sequence)

        self.data_columns = tuple(data_sequence_feature_space[0].columns)

        # make sure data is in an pomegranate compatible format!
        data_train_sequence = [
            np.ascontiguousarray(feature_data[list(self.data_columns)].to_numpy().copy())
            for feature_data in data_sequence_feature_space
        ]

        _, history = new_model.fit(
            sequences=data_train_sequence,
            labels=labels_train_sequence_str.copy(),
            algorithm=self.algo_train,
            stop_threshold=self.stop_threshold,
            max_iterations=self.max_iterations,
            return_history=True,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            multiple_check_input=False,
        )

        new_model.name = self.name

        self.model = new_model

        return (
            self,
            {"self": history, "transition_model": transition_model_history, "stride_model": stride_model_history},
        )
