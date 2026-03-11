"""Segmentation _model base classes and helper."""

from __future__ import annotations

# ruff: noqa: UP045
from collections.abc import Sequence
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import tpcp
from tpcp import OptiPara, cf, make_optimize_safe
from typing_extensions import Self

from gaitmap.base import _BaseSerializable
from gaitmap.utils.datatype_helper import (
    SingleSensorData,
    SingleSensorRegionsOfInterestList,
)
from gaitmap_mad.stride_segmentation.hmm._backend import BaseHmmBackend, BaseTrainableHmm, get_default_hmm_backend
from gaitmap_mad.stride_segmentation.hmm._config import CompositeHmmConfig, HmmSubModelConfig, RothHmmConfig
from gaitmap_mad.stride_segmentation.hmm._hmm_feature_transform import RothHmmFeatureTransformer
from gaitmap_mad.stride_segmentation.hmm._state import HMMState
from gaitmap_mad.stride_segmentation.hmm.legacy._utils import ShortenedHMMPrint
from gaitmap_mad.stride_segmentation.hmm._utils import (
    _DataToShortError,
    convert_region_list_to_transition_list,
    get_train_data_sequences_regions,
    get_train_data_sequences_transitions,
    validate_trainable_region_list,
)

DEFAULT_HMM_BACKEND = get_default_hmm_backend()


def create_fully_labeled_hidden_state_sequences(
    data_train_sequence: Sequence[pd.DataFrame],
    region_list_sequence: Sequence[SingleSensorRegionsOfInterestList],
    module_models: dict[str, Any],
    state_offsets: dict[str, int],
    transition_model_name: str,
    algo_predict: Literal["viterbi", "map"],
):
    """Create fully labeled hidden-state sequences from typed regions and trained submodels.

    Parameters
    ----------
    data_train_sequence
        Sequence of feature-space datasets.
    region_list_sequence
        Sequence of typed region lists with `start`, `end`, and `type`.
        `type` values are expected to refer to the configured explicit module names.
        Regions covered by no explicit module are treated as belonging to the implicit transition module.
    module_models
        Trained per-module models keyed by module name.
    state_offsets
        State-index offsets of each module in the combined final model.
    transition_model_name
        Name of the module that models the implicit transition regions.
    algo_predict
        Prediction algorithm used to create state labels from the trained submodels.

    Returns
    -------
    list of np.ndarray
        Fully labeled hidden-state sequences aligned with `data_train_sequence`.

    """
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
                    + state_offsets[region_type]
                )
            except _DataToShortError:
                continue

        labels_train_sequence.append(labels_train)

    return labels_train_sequence


def _get_training_sequences_for_module(
    module_config: HmmSubModelConfig,
    model_config: CompositeHmmConfig,
    data_sequence_feature_space: list[pd.DataFrame],
    region_list_feature_space: list[SingleSensorRegionsOfInterestList],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    module_name = module_config.name
    if module_name == model_config.transition_model_name:
        train_sequence, init_state_labels = get_train_data_sequences_transitions(
            data_sequence_feature_space, region_list_feature_space, module_config.n_states
        )
        if len(train_sequence) == 0:
            raise ValueError(
                "The configured transition module did not receive any trainable data. "
                "Either no implicit transition regions were found or all transition regions became too short after "
                "feature transformation."
            )
        return train_sequence, init_state_labels

    train_sequence, init_state_labels = get_train_data_sequences_regions(
        data_sequence_feature_space,
        region_list_feature_space,
        region_type=module_name,
        n_states=module_config.n_states,
    )
    if len(train_sequence) == 0:
        raise ValueError(
            f"The configured submodule `{module_name}` did not receive any trainable regions. "
            "Ensure that the region lists contain this type and that the regions remain long enough after feature "
            "transformation."
        )
    return train_sequence, init_state_labels


def _predict_labeled_training_sequences(
    data_sequence_feature_space: list[pd.DataFrame],
    region_list_feature_space: list[SingleSensorRegionsOfInterestList],
    trained_models: dict[str, Any],
    module_offsets: dict[str, int],
    transition_model_name: str,
    algo_predict: Literal["viterbi", "map"],
) -> list[np.ndarray]:
    return create_fully_labeled_hidden_state_sequences(
        data_sequence_feature_space,
        region_list_feature_space,
        trained_models,
        module_offsets,
        transition_model_name,
        algo_predict,
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


class RothSegmentationHmm(BaseSegmentationHmm, ShortenedHMMPrint):
    """A hierarchical HMM model for stride segmentation proposed by Roth et al. [1]_.

    This model uses individually trained HMM submodules that are combined into one final segmentation HMM.
    One submodule is reserved for the implicit transition regions, while all other submodules are trained on explicit
    typed regions.
    A final model is created by combining the transition matrices of the trained submodules and allowing transitions
    between the higher-level states where they occur in the labeled data.

    Parameters
    ----------
    hmm_config
        Serializable configuration bundle containing the HMM submodel topology, feature extraction parameters, and
        training/prediction settings.
    model
        The serialized trained HMM state.
        This can be set to `None` initially.
        A trained state will then be created during the optimization step.
        If you want to use a pre-trained model, you can set this parameter to the respective model state.
    backend
        Backend implementation that provides the backend-specific HMM primitives used for prediction and the final
        combined-model training step.

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
    The public trained-model parameter is stored as a serializable `HMMState`.
    The default backend in this refactor step is `PomegranateHmmBackend`.

    References
    ----------
    .. [1] Roth, N., Küderle, A., Ullrich, M. et al. Hidden Markov Model based stride segmentation on unsupervised
           free-living gait data in Parkinson`s disease patients. J NeuroEngineering Rehabil 18, 93 (2021).
           https://doi.org/10.1186/s12984-021-00883-7

    """

    hmm_config: RothHmmConfig
    model: OptiPara[Optional[HMMState]]
    backend: BaseHmmBackend

    feature_space_data_: pd.DataFrame
    hidden_state_sequence_feature_space_: np.ndarray

    @classmethod
    def _from_json_dict(cls, json_dict: dict) -> Self:
        params = json_dict["params"].copy()
        if "hmm_config" not in params and "model_config" in params:
            params["hmm_config"] = RothHmmConfig(
                model_config=params.pop("model_config"),
                feature_transform=params.pop("feature_transform", RothHmmFeatureTransformer()),
                algo_predict=params.pop("algo_predict", "viterbi"),
                algo_train=params.pop("algo_train", "baum-welch"),
                stop_threshold=params.pop("stop_threshold", 1e-9),
                max_iterations=params.pop("max_iterations", 1),
                initialization=params.pop("initialization", "labels"),
                verbose=params.pop("verbose", True),
                n_jobs=params.pop("n_jobs", 1),
                name=params.pop("name", "segmentation_model"),
            )
        input_data = {k: params[k] for k in tpcp.get_param_names(cls) if k in params}
        return cls(**input_data)

    def _to_json_dict(self) -> dict[str, Any]:
        return {
            "_gaitmap_obj": self.__class__.__name__,
            "params": {
                "hmm_config": self.hmm_config,
                "model": self.model,
            },
        }

    def __init__(
        self,
        hmm_config: RothHmmConfig = cf(RothHmmConfig()),
        model: Optional[HMMState] = None,
        backend: BaseHmmBackend = cf(DEFAULT_HMM_BACKEND),
    ) -> None:
        self.hmm_config = hmm_config
        self.model = model
        self.backend = backend

    @property
    def model_config(self) -> CompositeHmmConfig:
        return self.hmm_config.model_config

    @property
    def feature_transform(self) -> RothHmmFeatureTransformer:
        return self.hmm_config.feature_transform

    @property
    def algo_predict(self) -> Literal["viterbi", "map"]:
        return self.hmm_config.algo_predict

    @property
    def algo_train(self) -> Literal["viterbi", "baum-welch"]:
        return self.hmm_config.algo_train

    @property
    def stop_threshold(self) -> float:
        return self.hmm_config.stop_threshold

    @property
    def max_iterations(self) -> int:
        return self.hmm_config.max_iterations

    @property
    def initialization(self) -> Literal["labels", "fully-connected"]:
        return self.hmm_config.initialization

    @property
    def verbose(self) -> bool:
        return self.hmm_config.verbose

    @property
    def n_jobs(self) -> int:
        return self.hmm_config.n_jobs

    @property
    def name(self) -> str:
        return self.hmm_config.name

    @property
    def data_columns(self) -> tuple[str, ...]:
        return self.feature_transform.transformed_feature_columns

    @property
    def n_states(self) -> int:
        """Return the number of states of the final model."""
        return sum(module.n_states for module in self.model_config.modules)

    @property
    def _module_offsets(self) -> dict[str, int]:
        """Return the state offsets of each configured submodule in the combined model."""
        offsets = {}
        current_offset = 0
        for module in self.model_config.modules:
            offsets[module.name] = current_offset
            current_offset += module.n_states
        return offsets

    @property
    def stride_states(self) -> list[int]:
        """Return the ids of all stride-like states."""
        stride_states = []
        for module in self.model_config.modules:
            if module.role != "stride":
                continue
            stride_states.extend((np.arange(module.n_states) + self._module_offsets[module.name]).tolist())
        return stride_states

    @property
    def transition_states(self) -> list[int]:
        """Return the ids of the transition states."""
        transition_module = self.model_config.transition_model
        transition_offset = self._module_offsets[self.model_config.transition_model_name]
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

        self.hidden_state_sequence_feature_space_ = self.backend.predict(
            self.model,
            feature_data,
            expected_columns=self.data_columns,
            algorithm=self.algo_predict,
            verbose=self.verbose,
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

        This will first apply the feature transformation to the given data and then train the HMM model in three
        stages:

        1. Train each explicit typed-region submodule on its corresponding training regions.
        2. Train the implicit transition module on all uncovered regions between explicit regions.
        3. Assemble the final model by combining all trained submodules and train it for a couple further iterations.

        Parameters
        ----------
        data_sequence
            Sequence of gaitmap sensordata objects.
        region_list_sequence
            Sequence of typed region lists. Each list must have `start`, `end`, and `type` columns and a valid ROI/GS
            id column or index (`roi_id`/`gs_id`).
            `type` must only contain names of explicit modules configured in `model_config`.
            Regions must not overlap. Samples not covered by an explicit region are treated as the implicit transition
            region.
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
    ) -> tuple[Self, dict[str, Any]]:
        """Create and train the HMM model based on the given data and labels.

        This is identical to `self_optimize`, but returns additional information about the training process.
        The dictionary returned as second parameter contains the training history for each trained submodule and the
        combined final model `"self"`.

        Parameters
        ----------
        data_sequence
            Sequence of gaitmap sensordata objects.
        region_list_sequence
            Sequence of typed region lists. Each list must have `start`, `end`, and `type` columns and a valid ROI/GS
            id column or index (`roi_id`/`gs_id`).
            `type` must only contain names of explicit modules configured in `model_config`.
            Regions must not overlap. Samples not covered by an explicit region are treated as the implicit transition
            region.
            The number of region lists must match the number of sensordata objects (i.e. they must belong together).
        sampling_rate_hz
            Sampling frequency of the data.

        Returns
        -------
        self
            The trained model instance.
        history
            Dictionary containing the training history for each trained submodule and the final combined model.

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

        trained_models: dict[str, BaseTrainableHmm] = {}
        histories: dict[str, Any] = {}
        for module_config in self.model_config.modules:
            module_name = module_config.name
            train_sequence, init_state_labels = _get_training_sequences_for_module(
                module_config,
                self.model_config,
                data_sequence_feature_space,
                region_list_feature_space,
            )

            trained_model, history = self.backend.create_submodel(module_config).self_optimize_with_info(
                train_sequence, init_state_labels
            )
            trained_models[module_name] = trained_model
            histories[module_name] = history

        # predict hidden state labels for complete walking bouts
        module_offsets = self._module_offsets
        labels_train_sequence = _predict_labeled_training_sequences(
            data_sequence_feature_space,
            region_list_feature_space,
            trained_models,
            module_offsets,
            self.model_config.transition_model_name,
            self.algo_predict,
        )

        self.model, history = self.backend.finalize_model(
            trained_models=trained_models,
            labels_train_sequence=labels_train_sequence,
            data_sequence_feature_space=data_sequence_feature_space,
            data_columns=self.data_columns,
            model_config=self.model_config,
            module_offsets=module_offsets,
            initialization=self.initialization,
            algo_train=self.algo_train,
            stop_threshold=self.stop_threshold,
            max_iterations=self.max_iterations,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            name=self.name,
        )

        histories["self"] = history
        return self, histories
