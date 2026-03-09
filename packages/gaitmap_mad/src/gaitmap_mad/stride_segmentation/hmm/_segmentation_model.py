"""Segmentation _model base classes and helper."""

import copy
from collections.abc import Sequence
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import pomegranate as pg
import tpcp
from pomegranate import HiddenMarkovModel as pgHMM
from pomegranate.hmm import History
from tpcp import OptiPara, cf, make_optimize_safe
from typing_extensions import Self

from gaitmap.base import _BaseSerializable
from gaitmap.utils.datatype_helper import (
    SingleSensorData,
    SingleSensorRegionsOfInterestList,
    SingleSensorStrideList,
)
from gaitmap_mad.stride_segmentation.hmm._config import CompositeHmmConfig, HmmSubModelConfig
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
    check_history_for_training_failure,
    convert_region_list_to_transition_list,
    create_transition_matrix_fully_connected,
    extract_transitions_starts_stops_from_hidden_state_sequence,
    fix_model_names,
    get_model_distributions,
    get_train_data_sequences_regions,
    get_train_data_sequences_transitions,
    labels_to_strings,
    predict,
    validate_trainable_region_list,
)


def create_fully_labeled_gait_sequences(
    data_train_sequence: Sequence[pd.DataFrame],
    region_list_sequence: Sequence[SingleSensorRegionsOfInterestList],
    module_models: dict[str, SimpleHmm],
    module_offsets: dict[str, int],
    transition_model_name: str,
    algo_predict: Literal["viterbi", "map"],
):
    """Create fully labeled gait sequences from typed regions and trained submodels."""
    labels_train_sequence = []

    transition_model = module_models[transition_model_name]
    for data, region_list in zip(data_train_sequence, region_list_sequence):
        labels_train = np.zeros(len(data))

        transition_start_end_list = convert_region_list_to_transition_list(region_list, data.shape[0])
        for start, end in transition_start_end_list[["start", "end"]].to_numpy():
            transition_data_train = data[start:end]
            try:
                labels_train[start:end] = transition_model.predict_hidden_state_sequence(
                    transition_data_train, algorithm=algo_predict
                )
            except _DataToShortError:
                continue

        for start, end, region_type in region_list[["start", "end", "type"]].to_numpy():
            region_model = module_models[region_type]
            region_data_train = data[start:end]
            try:
                labels_train[start:end] = (
                    region_model.predict_hidden_state_sequence(region_data_train, algorithm=algo_predict)
                    + module_offsets[region_type]
                )
            except _DataToShortError:
                continue

        labels_train_sequence.append(labels_train)

    return labels_train_sequence


def _create_simple_hmm_from_config(config: HmmSubModelConfig) -> SimpleHmm:
    return SimpleHmm(
        n_states=config.n_states,
        n_gmm_components=config.n_gmm_components,
        architecture=config.architecture,
        algo_train=config.algo_train,
        stop_threshold=config.stop_threshold,
        max_iterations=config.max_iterations,
        verbose=config.verbose,
        n_jobs=config.n_jobs,
        name=config.name,
    )


def _coerce_region_list_input(
    region_list: pd.DataFrame, explicit_region_model_names: tuple[str, ...]
) -> SingleSensorRegionsOfInterestList:
    """Normalize legacy stride-list input to the new typed-region format when possible."""
    if "type" in region_list.reset_index().columns:
        return region_list
    if len(explicit_region_model_names) != 1:
        raise ValueError(
            "The provided training regions do not contain a `type` column. "
            "Automatic conversion from the legacy stride-list format is only possible when exactly one explicit "
            f"region module exists. Got explicit modules: {list(explicit_region_model_names)}"
        )
    coerced_region_list = region_list.reset_index().copy()
    if not {"roi_id", "gs_id"} & set(coerced_region_list.columns):
        coerced_region_list.insert(0, "roi_id", np.arange(len(coerced_region_list)))
    coerced_region_list["type"] = explicit_region_model_names[0]
    return coerced_region_list


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
    def stride_states(self) -> list[int]:
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
        region_list_sequence: Sequence[SingleSensorRegionsOfInterestList],
        sampling_rate_hz: float,
    ) -> Self:
        """Create and train the HMM model based on the given data and labels.

        Parameters
        ----------
        data_sequence
            Sequence of gaitmap sensordata objects.
        region_list_sequence
            Sequence of typed region lists.
            The number of region lists must match the number of sensordata objects (i.e. they must belong together).
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
        region_list_sequence: Sequence[SingleSensorRegionsOfInterestList],
        sampling_rate_hz: float,
    ) -> tuple[Self, Any]:
        """Create and train the HMM model based on the given data and labels.

        This is identical to `self_optimize`, but can return additional information about the training process.

        Parameters
        ----------
        data_sequence
            Sequence of gaitmap sensordata objects.
        region_list_sequence
            Sequence of typed region lists.
            The number of region lists must match the number of sensordata objects (i.e. they must belong together).
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
    """A hierarchical HMM model for stride segmentation proposed by Roth et al. [1]_.

    This model differentiates between strides and transitions.
    Both data sections are modeled by individual HMMs and are trained separately.
    A final model is created by combining the transition matrices of the two models and allowing for transitions between
    these higher level states at the start or end of a stride.

    Parameters
    ----------
    model_config
        The configuration of the named HMM submodules that are trained and combined into the final model.
        One module is interpreted as the implicit transition model and all remaining modules are expected to be covered
        by typed input regions during optimization.
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
        If you want to adjust the stop threshold for the individual submodels, do so via the corresponding entries in
        `model_config.modules`.
    max_iterations
        The maximum number of iterations to perform during the optimization.
        Note, that this is the value for the "combined" training of the final model.
        We recommend keeping this value at 1, as the combined model training only adjusts the transition matrix.
        If you want to adjust the max iterations for the individual submodels, do so via the corresponding entries in
        `model_config.modules`.
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

    Notes
    -----
    The final model is still stored as a raw `pomegranate` HMM in this step of the refactor.
    The `model_config` only replaces the dedicated submodel constructor inputs.

    References
    ----------
    .. [1] Roth, N., Küderle, A., Ullrich, M. et al. Hidden Markov Model based stride segmentation on unsupervised
           free-living gait data in Parkinson`s disease patients. J NeuroEngineering Rehabil 18, 93 (2021).
           https://doi.org/10.1186/s12984-021-00883-7

    """

    model_config: CompositeHmmConfig
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
    data_columns: OptiPara[Optional[tuple[str, ...]]]

    feature_space_data_: pd.DataFrame
    hidden_state_sequence_feature_space_: np.ndarray

    @classmethod
    def _from_json_dict(cls, json_dict: dict) -> Self:
        params = json_dict["params"].copy()
        if "model_config" not in params and {"stride_model", "transition_model"} <= set(params):
            stride_model = params.pop("stride_model")
            transition_model = params.pop("transition_model")
            stride_params = stride_model.get_params(deep=False) if isinstance(stride_model, SimpleHmm) else stride_model["params"]
            transition_params = (
                transition_model.get_params(deep=False)
                if isinstance(transition_model, SimpleHmm)
                else transition_model["params"]
            )
            params["model_config"] = CompositeHmmConfig(
                modules={
                    "transition": HmmSubModelConfig(
                        name="transition",
                        role="transition",
                        n_states=transition_params["n_states"],
                        n_gmm_components=transition_params["n_gmm_components"],
                        architecture=transition_params["architecture"],
                        algo_train=transition_params["algo_train"],
                        stop_threshold=transition_params["stop_threshold"],
                        max_iterations=transition_params["max_iterations"],
                        verbose=transition_params.get("verbose", True),
                        n_jobs=transition_params.get("n_jobs", 1),
                    ),
                    "stride": HmmSubModelConfig(
                        name="stride",
                        role="stride",
                        n_states=stride_params["n_states"],
                        n_gmm_components=stride_params["n_gmm_components"],
                        architecture=stride_params["architecture"],
                        algo_train=stride_params["algo_train"],
                        stop_threshold=stride_params["stop_threshold"],
                        max_iterations=stride_params["max_iterations"],
                        verbose=stride_params.get("verbose", True),
                        n_jobs=stride_params.get("n_jobs", 1),
                    ),
                }
            )
        input_data = {k: params[k] for k in tpcp.get_param_names(cls) if k in params}
        return cls(**input_data)

    def __init__(
        self,
        model_config: CompositeHmmConfig = cf(CompositeHmmConfig()),
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
        data_columns: Optional[tuple[str, ...]] = None,
    ) -> None:
        self.model_config = model_config
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
        return sum(module.n_states for module in self.model_config.modules.values())

    @property
    def module_offsets(self) -> dict[str, int]:
        """Return the state offsets of each configured submodule in the combined model."""
        offsets = {}
        current_offset = 0
        for name, module in self.model_config.modules.items():
            offsets[name] = current_offset
            current_offset += module.n_states
        return offsets

    @property
    def stride_states(self) -> list[int]:
        """Return the ids of all stride-like states."""
        stride_states = []
        for name, module in self.model_config.modules.items():
            if module.role != "stride":
                continue
            stride_states.extend((np.arange(module.n_states) + self.module_offsets[name]).tolist())
        return stride_states

    @property
    def transition_states(self) -> list[int]:
        """Return the ids of the transition states."""
        transition_module = self.model_config.transition_model
        transition_offset = self.module_offsets[self.model_config.transition_model_name]
        return (np.arange(transition_module.n_states) + transition_offset).tolist()

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
            self.hidden_state_sequence_feature_space_, data=data
        )
        return self

    def _transform(
        self,
        data_sequence: Sequence[pd.DataFrame],
        region_list_sequence: Optional[Sequence[pd.DataFrame]],
        sampling_rate_hz: float,
    ):
        """Perform feature transformation."""
        feature_transform = self.feature_transform.clone()

        data_sequence_feature_space = [
            feature_transform.transform(dataset, sampling_rate_hz=sampling_rate_hz).transformed_data_
            for dataset in data_sequence
        ]

        region_list_feature_space = None
        if region_list_sequence:
            region_list_feature_space = [
                feature_transform.transform(
                    roi_list=region_list, sampling_rate_hz=sampling_rate_hz
                ).transformed_roi_list_
                for region_list in region_list_sequence
            ]
        return data_sequence_feature_space, region_list_feature_space

    def self_optimize(
        self,
        data_sequence: Sequence[SingleSensorData],
        region_list_sequence: Sequence[SingleSensorRegionsOfInterestList],
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
        region_list_sequence
            Sequence of typed region lists with `start`, `end`, and `type`.
            The number of region lists must match the number of sensordata objects (i.e. they must belong together).
        sampling_rate_hz
            Sampling frequency of the data.

        Returns
        -------
        self
            The trained model instance.

        """
        return self.self_optimize_with_info(data_sequence, region_list_sequence, sampling_rate_hz=sampling_rate_hz)[0]

    @make_optimize_safe
    def self_optimize_with_info(
        self,
        data_sequence: Sequence[SingleSensorData],
        region_list_sequence: Sequence[SingleSensorRegionsOfInterestList],
        sampling_rate_hz: float,
    ) -> tuple[Self, dict[str, History]]:
        """Create and train the HMM model based on the given data and labels.

        This is identical to `self_optimize`, but returns additional information about the training process.
        The dictionary returned as second parameter contains the training history for each of the three models (
        stride-model, transition-model, and the combined final model "self").

        Parameters
        ----------
        data_sequence
            Sequence of gaitmap sensordata objects.
        region_list_sequence
            Sequence of typed region lists with `start`, `end`, and `type`.
            The number of region lists must match the number of sensordata objects (i.e. they must belong together).
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

        validated_region_list_sequence = [
            validate_trainable_region_list(
                _coerce_region_list_input(region_list, self.model_config.explicit_region_model_names),
                self.model_config.explicit_region_model_names,
            )
            for region_list in region_list_sequence
        ]

        # perform feature transformation
        data_sequence_feature_space, region_list_feature_space = self._transform(
            data_sequence, validated_region_list_sequence, sampling_rate_hz
        )
        if region_list_feature_space is None:
            raise RuntimeError("The feature transform did not produce region lists for optimization.")

        trained_models: dict[str, SimpleHmm] = {}
        histories: dict[str, History] = {}
        for module_name, module_config in self.model_config.modules.items():
            if module_name == self.model_config.transition_model_name:
                train_sequence, init_state_labels = get_train_data_sequences_transitions(
                    data_sequence_feature_space, region_list_feature_space, module_config.n_states
                )
            else:
                train_sequence, init_state_labels = get_train_data_sequences_regions(
                    data_sequence_feature_space,
                    region_list_feature_space,
                    region_type=module_name,
                    n_states=module_config.n_states,
                )

            trained_model, history = _create_simple_hmm_from_config(module_config).self_optimize_with_info(
                train_sequence, init_state_labels
            )
            trained_models[module_name] = trained_model
            histories[module_name] = history

        # For model combination actually only the transition probabilities will be updated, while keeping the already
        # learned distributions for all states. This can be achieved by "labeled" training, where basically just the
        # number of transitions will be counted.
        distributions = []
        for module_name in self.model_config.modules:
            distributions.extend(get_model_distributions(trained_models[module_name].model))

        # predict hidden state labels for complete walking bouts
        module_offsets = self.module_offsets
        labels_train_sequence = create_fully_labeled_gait_sequences(
            data_sequence_feature_space,
            region_list_feature_space,
            trained_models,
            module_offsets,
            self.model_config.transition_model_name,
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
            trans_mat = np.zeros((self.n_states, self.n_states))
            for module_name, module_config in self.model_config.modules.items():
                module_transition_matrix = trained_models[module_name].model.dense_transition_matrix()[:-2, :-2]
                offset = module_offsets[module_name]
                trans_mat[
                    offset : offset + module_config.n_states,
                    offset : offset + module_config.n_states,
                ] = module_transition_matrix

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
            sequences=np.array(data_train_sequence, dtype=object),
            labels=np.array(labels_train_sequence_str, dtype=object).copy(),
            algorithm=self.algo_train,
            stop_threshold=self.stop_threshold,
            max_iterations=self.max_iterations,
            return_history=True,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            multiple_check_input=False,
        )
        check_history_for_training_failure(history)

        new_model.name = self.name

        self.model = new_model

        histories["self"] = history
        return self, histories
