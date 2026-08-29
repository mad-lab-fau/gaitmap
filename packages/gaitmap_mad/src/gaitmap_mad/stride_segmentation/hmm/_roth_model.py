"""Roth hidden Markov model for stride segmentation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from importlib.resources import files

import pandas as pd
from tpcp import cf, make_optimize_safe
from typing_extensions import Self

from gaitmap.data_transform import ButterworthFilter
from gaitmap.utils.datatype_helper import SingleSensorData, SingleSensorStrideList
from gaitmap_mad.stride_segmentation.hmm._composite_model import CompositeHmm
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
        routes={"transition": ("stride",), "stride": ("transition", "stride")},
        starts=("transition", "stride"),
        ends=("transition", "stride"),
    )


def _stride_borders_to_regions(n_samples: int, stride_list: SingleSensorStrideList) -> pd.DataFrame:
    regions = []
    cursor = 0
    for start, end in stride_list[["start", "end"]].to_numpy(dtype=int):
        if start < cursor or end <= start or end > n_samples:
            raise ValueError("Stride borders must be sorted, non-overlapping, and contained in the training data.")
        if start > cursor:
            regions.append((cursor, start, "transition"))
        regions.append((start, end, "stride"))
        cursor = end
    if cursor < n_samples:
        regions.append((cursor, n_samples, "transition"))
    return pd.DataFrame(regions, columns=["start", "end", "model"])


class RothSegmentationHmm(CompositeHmm):
    """Roth feature and annotation preset for a generic composite HMM.

    Stride borders are converted to complete ``stride``/``transition`` region
    tables before delegating to :class:`CompositeHmm`. The model topology stays
    configurable; the default merely captures the Roth architecture.

    Parameters
    ----------
    model
        Unfitted topology for training or a fitted backend-neutral model for
        inference.
    feature_transform
        Feature pipeline applied before HMM training and inference.
    inference_backend
        Decoder used for submodel orchestration and prediction.
    training_backend
        Atomic backend used for independent and final transition fitting.

    Attributes
    ----------
    feature_space_data_
        Transformed data passed to the inference backend.
    hidden_state_sequence_feature_space_
        Hidden states at the feature sampling rate.
    hidden_state_sequence_
        Hidden states resampled to the input data.

    Other Parameters
    ----------------
    data
        Single-sensor data passed to :meth:`predict`.
    sampling_rate_hz
        Sampling rate of the input data.

    """

    def __init__(
        self,
        model: HmmModel = cf(_default_roth_topology()),
        feature_transform: BaseHmmFeatureTransformer = cf(RothHmmFeatureTransformer()),
        inference_backend: ScipyHmmInference = cf(ScipyHmmInference()),
        training_backend: PomegranateHmmTrainer = cf(PomegranateHmmTrainer()),
    ) -> None:
        super().__init__(model, feature_transform, inference_backend, training_backend)

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
        return self.model.stride_states

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

        regions = [
            _stride_borders_to_regions(len(data), stride_list)
            for data, stride_list in zip(data_sequence, stride_list_sequence)
        ]
        return self._self_optimize_composite(data_sequence, regions, sampling_rate_hz=sampling_rate_hz)
