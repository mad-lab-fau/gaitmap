"""Behavioral tests for the Roth HMM segmentation pipeline."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from gaitmap.data_transform import SlidingWindowMean
from gaitmap.stride_segmentation.hmm import (
    HmmStrideSegmentation,
    RothHmmFeatureTransformer,
    RothSegmentationHmm,
)
from gaitmap.utils.coordinate_conversion import convert_left_foot_to_fbf, convert_to_fbf
from gaitmap.utils.datatype_helper import is_multi_sensor_stride_list, is_single_sensor_stride_list
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin


class TestMetaFunctionalityRothSegmentationHmm(TestAlgorithmMixin):
    __test__ = True
    algorithm_class = RothSegmentationHmm

    @pytest.fixture
    def after_action_instance(self, healthy_example_imu_data) -> RothSegmentationHmm:
        data = convert_left_foot_to_fbf(healthy_example_imu_data["left_sensor"])
        return RothSegmentationHmm.from_pretrained().predict(data, sampling_rate_hz=204.8)


class TestMetaFunctionalityHmmStrideSegmentation(TestAlgorithmMixin):
    __test__ = True
    algorithm_class = HmmStrideSegmentation

    @pytest.fixture
    def after_action_instance(self, healthy_example_imu_data) -> HmmStrideSegmentation:
        data = convert_left_foot_to_fbf(healthy_example_imu_data["left_sensor"])
        return HmmStrideSegmentation().segment(data, sampling_rate_hz=204.8)


class TestMetaFunctionalityRothHmmFeatureTransformer(TestAlgorithmMixin):
    __test__ = True
    algorithm_class = RothHmmFeatureTransformer

    @pytest.fixture
    def after_action_instance(
        self, healthy_example_imu_data, healthy_example_stride_borders
    ) -> RothHmmFeatureTransformer:
        data = convert_left_foot_to_fbf(healthy_example_imu_data["left_sensor"])
        return RothHmmFeatureTransformer().transform(
            data,
            roi_list=healthy_example_stride_borders["left_sensor"],
            sampling_rate_hz=204.8,
        )


class TestRothHmmFeatureTransformer:
    @pytest.mark.parametrize("target_sampling_rate", [50, 25, 16.3])
    def test_inverse_transform_preserves_values_and_target_length(self, target_sampling_rate) -> None:
        transform = RothHmmFeatureTransformer(sampling_rate_feature_space_hz=target_sampling_rate)
        states = np.array([0, 1, 2, 2, 4, 5])
        data = np.zeros(int(len(states) * np.round(100 / target_sampling_rate)))

        transformed_states = transform.inverse_transform_state_sequence(states, data=data)

        assert_array_equal(np.unique(transformed_states), np.unique(states))
        assert len(transformed_states) == len(data)

    @pytest.mark.parametrize("features", [["raw"], ["raw", "gradient"], ["raw", "gradient", "mean"]])
    @pytest.mark.parametrize("axes", [["gyr_ml"], ["acc_pa"], ["gyr_ml", "acc_pa"]])
    def test_select_features(self, features, healthy_example_imu_data, axes) -> None:
        data = convert_left_foot_to_fbf(healthy_example_imu_data["left_sensor"]).iloc[:100]
        transformed = (
            RothHmmFeatureTransformer(features=features, axes=axes)
            .transform(data, sampling_rate_hz=100)
            .transformed_data_
        )
        feature_suffixes = {"raw": "", "gradient": "__gradient", "mean": "__mean"}

        assert transformed.shape[1] == len(features) * len(axes)
        assert set(transformed.columns) == {
            f"{feature}{feature_suffixes[feature]}__{axis}" for feature in features for axis in axes
        }

    def test_features_without_resampling_or_standardization(self, healthy_example_imu_data) -> None:
        data = convert_left_foot_to_fbf(healthy_example_imu_data["left_sensor"]).iloc[:100]
        transform = RothHmmFeatureTransformer(
            sampling_rate_feature_space_hz=100,
            standardization=False,
            low_pass_filter=None,
            features=["raw", "mean"],
            axes=["gyr_ml", "acc_pa"],
        ).transform(data, sampling_rate_hz=100)

        assert_array_equal(transform.transformed_data_["raw__gyr_ml"], data["gyr_ml"])
        assert_array_equal(transform.transformed_data_["raw__acc_pa"], data["acc_pa"])
        for axis in ("gyr_ml", "acc_pa"):
            expected = (
                SlidingWindowMean(window_size_s=transform.window_size_s)
                .transform(data[axis], sampling_rate_hz=100)
                .transformed_data_
            )
            assert_array_equal(transform.transformed_data_[f"mean__mean__{axis}"], expected)

    def test_roi_resampling(self) -> None:
        roi = pd.DataFrame([[0, 100], [200, 300], [400, 500]], columns=["start", "end"])
        result = RothHmmFeatureTransformer(sampling_rate_feature_space_hz=50).transform(
            roi_list=roi, sampling_rate_hz=100
        )
        assert_array_equal(result.transformed_roi_list_, [[0, 50], [100, 150], [200, 250]])


class TestHmmStrideSegmentation:
    def test_single_sensor_results(self, healthy_example_imu_data) -> None:
        data = convert_left_foot_to_fbf(healthy_example_imu_data["left_sensor"])
        result = HmmStrideSegmentation().segment(data, sampling_rate_hz=204.8)

        assert is_single_sensor_stride_list(result.stride_list_)
        assert result.hidden_state_sequence_ is result.result_model_.hidden_state_sequence_
        assert_array_equal(result.matches_start_end_, result.stride_list_.to_numpy())

    def test_multi_sensor_results_are_independent(self, healthy_example_imu_data) -> None:
        data = convert_to_fbf(healthy_example_imu_data, left_like="left_", right_like="right_")
        result = HmmStrideSegmentation().segment(data, sampling_rate_hz=204.8)

        assert is_multi_sensor_stride_list(result.stride_list_)
        assert result.result_model_["left_sensor"] is not result.result_model_["right_sensor"]
        assert (
            result.hidden_state_sequence_["left_sensor"] is result.result_model_["left_sensor"].hidden_state_sequence_
        )
        assert (
            result.hidden_state_sequence_["right_sensor"] is result.result_model_["right_sensor"].hidden_state_sequence_
        )

    def test_border_refinement_can_be_disabled(self, healthy_example_imu_data) -> None:
        data = convert_left_foot_to_fbf(healthy_example_imu_data["left_sensor"])

        refined = HmmStrideSegmentation().segment(data, sampling_rate_hz=204.8)
        unrefined = HmmStrideSegmentation(snap_to_min_win_ms=None).segment(data, sampling_rate_hz=204.8)

        assert not np.array_equal(refined.matches_start_end_original_, refined.matches_start_end_)
        assert_array_equal(unrefined.matches_start_end_original_, unrefined.matches_start_end_)
