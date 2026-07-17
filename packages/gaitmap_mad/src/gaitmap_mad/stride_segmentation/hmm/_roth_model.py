"""Roth hidden Markov model for stride segmentation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from importlib.resources import files
from typing import Optional

import numpy as np
import pandas as pd
from tpcp import OptiPara, cf, make_action_safe, make_optimize_safe
from typing_extensions import Self

from gaitmap.base import BaseAlgorithm
from gaitmap.data_transform import ButterworthFilter
from gaitmap.utils.datatype_helper import SingleSensorData, SingleSensorStrideList
from gaitmap_mad.stride_segmentation.hmm._hmm_feature_transform import (
    BaseHmmFeatureTransformer,
    RothHmmFeatureTransformer,
)
from gaitmap_mad.stride_segmentation.hmm._model import HmmModel
from gaitmap_mad.stride_segmentation.hmm._pomegranate_backend import PomegranateHmmTrainer
from gaitmap_mad.stride_segmentation.hmm._scipy_inference import ScipyHmmInference


def _default_roth_topology() -> HmmModel:
    return HmmModel.compose(
        parts={
            "transition": HmmModel.left_right(n_states=5, n_gmm_components=3),
            "stride": HmmModel.left_right(n_states=20, n_gmm_components=3, cycle=True),
        },
        routes={"transition": ("stride",), "stride": ("transition",)},
        starts=("transition",),
        ends=("transition",),
    )


def _equidistant_state_labels(length: int, states: tuple[int, ...]) -> np.ndarray:
    if length < len(states):
        raise ValueError(f"A labeled region with {length} samples is too short for {len(states)} HMM states.")
    return np.asarray(states)[np.minimum(np.arange(length) * len(states) // length, len(states) - 1)]


def _create_roth_state_labels(
    n_samples: int,
    stride_list: SingleSensorStrideList,
    transition_states: tuple[int, ...],
    stride_states: tuple[int, ...],
) -> np.ndarray:
    labels = np.empty(n_samples, dtype=int)
    cursor = 0
    for start, end in stride_list[["start", "end"]].to_numpy(dtype=int):
        if start < cursor or end <= start or end > n_samples:
            raise ValueError("Stride borders must be sorted, non-overlapping, and contained in the training data.")
        if start > cursor:
            labels[cursor:start] = _equidistant_state_labels(start - cursor, transition_states)
        labels[start:end] = _equidistant_state_labels(end - start, stride_states)
        cursor = end
    if cursor < n_samples:
        labels[cursor:] = _equidistant_state_labels(n_samples - cursor, transition_states)
    return labels


class RothSegmentationHmm(BaseAlgorithm):
    """Predict Roth stride-segmentation states using a fitted HMM.

    Parameters
    ----------
    model
        Backend-neutral fitted HMM parameters. Use :meth:`from_pretrained` to
        load the bundled FallRiskPD model.
    feature_transform
        Feature pipeline applied before HMM inference.
    inference_backend
        Decoder used to predict the most likely hidden-state sequence.
    training_backend
        Backend used by :meth:`self_optimize` to fit a compiled model.

    Attributes
    ----------
    feature_space_data_
        Transformed data passed to the inference backend.
    hidden_state_sequence_feature_space_
        Hidden-state sequence at the feature sampling rate.
    hidden_state_sequence_
        Hidden-state sequence resampled to the input data.

    Other Parameters
    ----------------
    data
        Single-sensor data passed to :meth:`predict`.
    sampling_rate_hz
        Sampling rate of the input data.

    """

    _action_methods = ("predict",)

    # tpcp evaluates this annotation on Python 3.9, where PEP 604 unions cannot be evaluated.
    model: OptiPara[Optional[HmmModel]]  # noqa: UP045
    feature_transform: BaseHmmFeatureTransformer
    inference_backend: ScipyHmmInference
    training_backend: PomegranateHmmTrainer

    feature_space_data_: pd.DataFrame
    hidden_state_sequence_feature_space_: np.ndarray
    hidden_state_sequence_: np.ndarray

    def __init__(
        self,
        model: HmmModel | None = cf(_default_roth_topology()),
        feature_transform: BaseHmmFeatureTransformer = cf(RothHmmFeatureTransformer()),
        inference_backend: ScipyHmmInference = cf(ScipyHmmInference()),
        training_backend: PomegranateHmmTrainer = cf(PomegranateHmmTrainer()),
    ) -> None:
        self.model = model
        self.feature_transform = feature_transform
        self.inference_backend = inference_backend
        self.training_backend = training_backend

    @classmethod
    def from_pretrained(cls) -> Self:
        """Load the bundled legacy FallRiskPD model."""
        model_json = (
            files("gaitmap_mad.stride_segmentation.hmm._pre_trained_models")
            .joinpath("fallriskpd_at_lab_model.json")
            .read_text(encoding="utf8")
        )
        return cls.from_legacy_json(model_json)

    @classmethod
    def from_legacy_json(cls, model_json: str) -> Self:
        """Load and compile a Roth model exported with pomegranate 0.14.

        This is an explicit one-way migration. The returned instance contains
        only :class:`HmmModel` arrays and works with every inference backend.
        """
        serialized = json.loads(model_json)
        try:
            params = serialized["params"]
            hmm_payload = params["model"]["hmm"]
            transition_states = int(params["transition_model"]["params"]["n_states"])
            stride_state_count = int(params["stride_model"]["params"]["n_states"])
            data_columns = tuple(params["data_columns"])
        except (KeyError, TypeError) as e:
            raise ValueError("The provided JSON is not a legacy RothSegmentationHmm export.") from e

        feature_params = params.get("feature_transform", {}).get("params", {})
        feature_kwargs = {
            key: feature_params[key]
            for key in ("axes", "features", "window_size_s", "standardization")
            if key in feature_params
        }
        sampling_rate = feature_params.get(
            "sampling_rate_feature_space_hz", feature_params.get("sampling_frequency_feature_space_hz")
        )
        if sampling_rate is not None:
            feature_kwargs["sampling_rate_feature_space_hz"] = sampling_rate
        if "low_pass_filter" in feature_params:
            serialized_filter = feature_params["low_pass_filter"]
            if serialized_filter is None:
                feature_kwargs["low_pass_filter"] = None
            else:
                filter_params = serialized_filter["params"]
                feature_kwargs["low_pass_filter"] = ButterworthFilter(
                    order=filter_params["order"],
                    cutoff_freq_hz=filter_params["cutoff_freq_hz"],
                    filter_type=filter_params.get("filter_type", filter_params.get("type", "lowpass")),
                )

        stride_states = tuple(range(transition_states, transition_states + stride_state_count))
        model = HmmModel.from_legacy_pomegranate(hmm_payload, data_columns=data_columns, stride_states=stride_states)
        return cls(model=model, feature_transform=RothHmmFeatureTransformer(**feature_kwargs))

    @property
    def stride_states(self) -> tuple[int, ...]:
        """Hidden states representing strides."""
        if self.model is None:
            raise ValueError("No trained model available.")
        return self.model.stride_states

    @make_action_safe
    def predict(self, data: SingleSensorData, sampling_rate_hz: float) -> Self:
        """Predict the hidden-state sequence for one sensor."""
        if self.model is None:
            raise ValueError("No trained model available. Call `self_optimize` or provide a fitted model.")

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        feature_transform = self.feature_transform.clone().transform(data, sampling_rate_hz=sampling_rate_hz)
        self.feature_space_data_ = feature_transform.transformed_data_
        self.hidden_state_sequence_feature_space_ = self.inference_backend.predict(self.model, self.feature_space_data_)
        self.hidden_state_sequence_ = feature_transform.inverse_transform_state_sequence(
            self.hidden_state_sequence_feature_space_, data=data
        )
        return self

    @make_optimize_safe
    def self_optimize(
        self,
        data_sequence: Sequence[SingleSensorData],
        stride_list_sequence: Sequence[SingleSensorStrideList],
        *,
        sampling_rate_hz: float,
    ) -> Self:
        """Fit the Roth model from continuous recordings and stride borders."""
        if not data_sequence or len(data_sequence) != len(stride_list_sequence):
            raise ValueError("Training data and stride lists must contain the same non-zero number of sequences.")

        feature_data = []
        state_sequences = []
        if self.model is None:
            raise ValueError("Training requires an unfitted HMM topology.")
        transition_states = self.model.state_indices("transition")
        stride_states = self.model.state_indices("stride")
        for data, stride_list in zip(data_sequence, stride_list_sequence):
            transformed = self.feature_transform.clone().transform(
                data, roi_list=stride_list, sampling_rate_hz=sampling_rate_hz
            )
            feature_data.append(transformed.transformed_data_)
            state_sequences.append(
                _create_roth_state_labels(
                    len(transformed.transformed_data_),
                    transformed.transformed_roi_list_,
                    transition_states=transition_states,
                    stride_states=stride_states,
                )
            )

        self.model = self.training_backend.fit(self.model, feature_data, state_sequences)
        return self
