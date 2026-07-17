"""Generic orchestration for independently trained, composed HMMs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from tpcp import OptiPara, cf, make_action_safe, make_optimize_safe
from typing_extensions import Self

from gaitmap.base import BaseAlgorithm
from gaitmap.utils.datatype_helper import SingleSensorData, SingleSensorRegionsOfInterestList
from gaitmap_mad.stride_segmentation.hmm._hmm_feature_transform import (
    BaseHmmFeatureTransformer,
    RothHmmFeatureTransformer,
)
from gaitmap_mad.stride_segmentation.hmm._model import HmmModel
from gaitmap_mad.stride_segmentation.hmm._pomegranate_backend import PomegranateHmmTrainer
from gaitmap_mad.stride_segmentation.hmm._scipy_inference import ScipyHmmInference


def _ordered_initialization_labels(length: int, topology: HmmModel) -> np.ndarray:
    n_states = topology.n_states
    if length < n_states:
        raise ValueError(f"A labelled region with {length} samples is too short for {n_states} HMM states.")
    return np.minimum(np.arange(length) * n_states // length, n_states - 1)


def _validate_ordered_initialization(topology: HmmModel) -> None:
    states = np.arange(topology.n_states)
    if (
        not topology.allowed_starts[0]
        or not topology.allowed_ends[-1]
        or not np.all(topology.allowed_transitions[states, states])
        or not np.all(topology.allowed_transitions[states[:-1], states[1:]])
    ):
        raise ValueError(
            "CompositeHmm requires every part to provide an ordered initialization path from its first to last state. "
            "Use the atomic trainer with caller-supplied labels for arbitrary state graphs."
        )


class CompositeHmm(BaseAlgorithm):
    """Train and predict with any number of named HMM parts.

    The training annotations must be complete region tables with ``start``,
    ``end``, and ``model`` columns. Every sample belongs to exactly one named
    part of the composed topology.
    """

    _action_methods = ("predict",)

    model: OptiPara[HmmModel]
    feature_transform: BaseHmmFeatureTransformer
    inference_backend: ScipyHmmInference
    training_backend: PomegranateHmmTrainer

    feature_space_data_: pd.DataFrame
    hidden_state_sequence_feature_space_: np.ndarray
    hidden_state_sequence_: np.ndarray

    def __init__(
        self,
        model: HmmModel,
        feature_transform: BaseHmmFeatureTransformer = cf(RothHmmFeatureTransformer()),
        inference_backend: ScipyHmmInference = cf(ScipyHmmInference()),
        training_backend: PomegranateHmmTrainer = cf(PomegranateHmmTrainer()),
    ) -> None:
        self.model = model
        self.feature_transform = feature_transform
        self.inference_backend = inference_backend
        self.training_backend = training_backend

    @make_action_safe
    def predict(self, data: SingleSensorData, sampling_rate_hz: float) -> Self:
        """Predict the hidden states of a fitted composite model."""
        if not self.model.is_fitted:
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
        region_list_sequence: Sequence[SingleSensorRegionsOfInterestList],
        *,
        sampling_rate_hz: float,
    ) -> Self:
        """Fit each named part, compose them, then adjust the final routing probabilities."""
        return self._self_optimize_composite(data_sequence, region_list_sequence, sampling_rate_hz=sampling_rate_hz)

    def _self_optimize_composite(  # noqa: C901
        self,
        data_sequence: Sequence[SingleSensorData],
        region_list_sequence: Sequence[SingleSensorRegionsOfInterestList],
        *,
        sampling_rate_hz: float,
    ) -> Self:
        if not data_sequence or len(data_sequence) != len(region_list_sequence):
            raise ValueError("Training data and region tables must contain the same non-zero number of sequences.")
        if self.model.composition is None:
            raise ValueError("Composite training requires a model created with `HmmModel.compose()`.")

        part_topologies = dict(self.model.composition.parts)
        for topology in part_topologies.values():
            _validate_ordered_initialization(topology)
        transformed_data = []
        transformed_regions = []
        for data, regions in zip(data_sequence, region_list_sequence):
            regions = self._validate_regions(regions, len(data), set(part_topologies))
            transformed = self.feature_transform.clone().transform(
                data, roi_list=regions, sampling_rate_hz=sampling_rate_hz
            )
            feature_regions = self._validate_regions(
                transformed.transformed_roi_list_, len(transformed.transformed_data_), set(part_topologies)
            )
            transformed_data.append(transformed.transformed_data_)
            transformed_regions.append(feature_regions)

        fitted_parts = {}
        for name, topology in part_topologies.items():
            observations = []
            labels = []
            for data, regions in zip(transformed_data, transformed_regions):
                for start, end in regions.loc[regions["model"] == name, ["start", "end"]].to_numpy(dtype=int):
                    observations.append(data.iloc[start:end])
                    labels.append(_ordered_initialization_labels(end - start, topology))
            if not observations:
                raise ValueError(f"No training region was labelled for model part {name!r}.")
            fitted_parts[name] = self.training_backend.fit(topology, observations, labels)

        composition = self.model.composition
        fitted_model = HmmModel.compose(
            fitted_parts,
            routes=dict(composition.routes),
            starts=composition.starts,
            ends=composition.ends,
        )
        state_positions = {state_id: position for position, state_id in enumerate(fitted_model.state_ids)}
        combined_labels = []
        for data, regions in zip(transformed_data, transformed_regions):
            labels = np.empty(len(data), dtype=int)
            for start, end, name in regions[["start", "end", "model"]].itertuples(index=False, name=None):
                part = fitted_parts[name]
                local_labels = self.inference_backend.predict(part, data.iloc[start:end])
                labels[start:end] = [state_positions[(name, *part.state_ids[state])] for state in local_labels]
            combined_labels.append(labels)

        self.model = self.training_backend.fit(fitted_model, transformed_data, combined_labels, train="transitions")
        return self

    @staticmethod
    def _validate_regions(regions: pd.DataFrame, n_samples: int, model_names: set[str]) -> pd.DataFrame:
        missing_columns = {"start", "end", "model"} - set(regions.columns)
        if missing_columns:
            raise ValueError(f"Region tables are missing required columns: {tuple(sorted(missing_columns))}.")
        if len(regions) == 0:
            raise ValueError("Region tables must be non-empty and cover the complete training sequence.")

        try:
            numeric_borders = regions[["start", "end"]].to_numpy(dtype=float)
        except (TypeError, ValueError) as e:
            raise ValueError("Region boundaries must be finite integer sample indices.") from e
        if np.any(~np.isfinite(numeric_borders)) or np.any(numeric_borders != np.floor(numeric_borders)):
            raise ValueError("Region boundaries must be finite integer sample indices.")
        borders = numeric_borders.astype(int)
        if borders[0, 0] != 0 or borders[-1, 1] != n_samples or np.any(borders[:, 1] <= borders[:, 0]):
            raise ValueError("Regions must be positive-length and cover the complete training sequence.")
        if np.any(borders[1:, 0] != borders[:-1, 1]):
            raise ValueError("Regions must be sorted, non-overlapping, and have no gaps.")
        if unknown := set(regions["model"]) - model_names:
            raise ValueError(f"Region tables reference unknown model parts: {tuple(sorted(unknown))}.")
        normalized = regions.copy()
        normalized["start"] = borders[:, 0]
        normalized["end"] = borders[:, 1]
        return normalized
