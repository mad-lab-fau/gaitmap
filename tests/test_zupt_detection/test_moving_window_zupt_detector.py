from contextlib import nullcontext
from typing import Union

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from gaitmap.utils.consts import BF_ACC, BF_GYR, SF_ACC, SF_COLS, SF_GYR
from gaitmap.utils.exceptions import ValidationError
from gaitmap.zupt_detection import AredZuptDetector, NormZuptDetector, ShoeZuptDetector
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin


class TestMetaFunctionalityNormZuptDetector(TestAlgorithmMixin):
    __test__ = True
    algorithm_class = NormZuptDetector

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data):
        data_left = healthy_example_imu_data["left_sensor"].iloc[:500]
        return NormZuptDetector().detect(data_left, sampling_rate_hz=204.8)


class TestMetaFunctionalityShoeZuptDetector(TestAlgorithmMixin):
    __test__ = True
    algorithm_class = ShoeZuptDetector

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data):
        data_left = healthy_example_imu_data["left_sensor"].iloc[:500]
        return ShoeZuptDetector().detect(data_left, sampling_rate_hz=204.8)


class TestMetaFunctionalityAredZuptDetector(TestAlgorithmMixin):
    __test__ = True
    algorithm_class = AredZuptDetector

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data):
        data_left = healthy_example_imu_data["left_sensor"].iloc[:500]
        return AredZuptDetector().detect(data_left, sampling_rate_hz=204.8)


class TestNormZuptDetector:
    """Test the function `sliding_window_view`."""

    algorithm_class: Union[type[NormZuptDetector], type[AredZuptDetector]]

    @pytest.fixture(params=(NormZuptDetector, AredZuptDetector), autouse=True)
    def get_algorithm_class(self, request) -> None:
        self.algorithm_class = request.param

    @pytest.mark.parametrize(("ws", "sr"), [(1, 1), (1, 2), (2, 1), (2.49, 1)])
    def test_error_window_to_small(self, healthy_example_imu_data, ws, sr) -> None:
        with pytest.raises(ValidationError, match=r".*The effective window size is smaller*"):
            self.algorithm_class(window_length_s=ws).detect(
                healthy_example_imu_data["left_sensor"], sampling_rate_hz=sr
            )

    @pytest.mark.parametrize(
        ("overlap1", "overlap2", "valid"),
        [
            (0.5, 0.5, False),
            (1, None, False),
            (-0.5, None, False),
            (0.5, None, True),
            (0, None, True),
            (None, 0, True),
            (None, 200, False),
            (None, 201, False),
            (None, 0.5, False),
            (None, 1, True),
            (None, -1, True),
            (None, 199, True),
        ],
    )
    def test_wrong_window_overlap(self, overlap1, overlap2, valid, healthy_example_imu_data) -> None:
        context = pytest.raises(ValidationError) if not valid else nullcontext()
        with context:
            self.algorithm_class(window_length_s=1, window_overlap=overlap1, window_overlap_samples=overlap2).detect(
                healthy_example_imu_data["left_sensor"], sampling_rate_hz=200
            )

    @pytest.mark.parametrize(("win_len", "overlap", "valid"), [(1, 0, True), (10 / 200, 0.99, False)])
    def test_effective_overlap_error(self, win_len, overlap, valid, healthy_example_imu_data) -> None:
        if not valid:
            with pytest.raises(ValidationError, match=r".*The effective window overlap after rounding is 1"):
                self.algorithm_class(
                    window_overlap=overlap, window_overlap_samples=None, window_length_s=win_len
                ).detect(healthy_example_imu_data["left_sensor"], sampling_rate_hz=200)
        else:
            self.algorithm_class(window_overlap=overlap, window_overlap_samples=None, window_length_s=win_len).detect(
                healthy_example_imu_data["left_sensor"], sampling_rate_hz=200
            )

    @pytest.mark.parametrize(
        ("sensor", "axis", "valid"),
        [
            ("acc", SF_ACC, True),
            ("acc", BF_ACC, True),
            ("acc", BF_GYR, False),
            ("gyr", BF_GYR, True),
            ("gyr", SF_GYR, True),
            ("gyr", SF_ACC, False),
        ],
    )
    def test_single_sensor_data(self, sensor, axis, valid) -> None:
        data = pd.DataFrame(np.empty((10, len(axis))), columns=axis)
        if valid is False:
            with pytest.raises(ValidationError):
                self.algorithm_class(sensor=sensor, window_length_s=0.5).detect(data, sampling_rate_hz=10)
        else:
            self.algorithm_class(sensor=sensor, window_length_s=0.5).detect(data, sampling_rate_hz=10)

    def test_debug_outputs(self) -> None:
        data = pd.DataFrame(np.empty((10, 3)), columns=BF_GYR)
        zupt = self.algorithm_class(
            sensor="gyr", window_length_s=0.5, window_overlap=0.2, window_overlap_samples=None
        ).detect(data, sampling_rate_hz=10)
        assert zupt.window_length_samples_ == 5
        assert zupt.window_overlap_samples_ == 1

    def test_invalid_input_metric_default_overlap(self, healthy_example_imu_data) -> None:
        """Test if value error is raised correctly on invalid input dimensions."""
        with pytest.raises(ValueError, match=r".*Invalid metric passed!.*"):
            self.algorithm_class(metric="invalid").detect(
                healthy_example_imu_data["left_sensor"], sampling_rate_hz=204.8
            )

    def test_invalid_window_length(self, healthy_example_imu_data) -> None:
        """Test if value error is raised when window longer than signal."""
        with pytest.raises(ValueError, match=r".*Invalid window length*"):
            self.algorithm_class(window_length_s=1000).detect(
                healthy_example_imu_data["left_sensor"].iloc[:500], sampling_rate_hz=1.0
            )

    @pytest.mark.parametrize(
        ("win_overlap_samples", "expected"),
        [
            (50, 50),
            (-1, 99),
            (-10, 90),
            (0, 0),
            (-100, 0),
            (1, 1),
        ],
    )
    def test_sample_win_overlap(self, win_overlap_samples, expected, healthy_example_imu_data) -> None:
        """Test that window overlap is correctly calculated when provided in samples."""
        out = self.algorithm_class(
            window_length_s=100, window_overlap_samples=win_overlap_samples, window_overlap=None
        ).detect(healthy_example_imu_data["left_sensor"].iloc[:500], sampling_rate_hz=1.0)
        assert out.window_overlap_samples_ == expected

    def test_single_window_fit(self) -> None:
        """Test input where only a single window length fits within input signal."""
        test_input = np.array([0, 0, 0, 0, 0, 1, 1, 1])
        test_input = pd.DataFrame(np.column_stack([test_input, test_input, test_input]), columns=SF_GYR)
        expected_output = np.array([1, 1, 1, 1, 1, 0, 0, 0])
        expected_output_sequence = pd.DataFrame([[0, 5]], columns=["start", "end"], dtype="Int64")
        window_length = 5
        test_output = self.algorithm_class(
            window_length_s=window_length, window_overlap=0, window_overlap_samples=None
        ).detect(test_input, sampling_rate_hz=1)
        assert_array_equal(test_output.per_sample_zupts_, expected_output)
        assert_frame_equal(test_output.zupts_, expected_output_sequence)
        assert test_output.min_vel_value_ == 0.0
        assert test_output.min_vel_index_ == 2

    def test_max_overlap_metric_max_w4_default_overlap(self) -> None:
        """Test binary input data on max metric with window size 4."""
        test_input = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
        test_input = pd.DataFrame(np.column_stack([test_input, test_input, test_input]), columns=SF_GYR)
        expected_output = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
        expected_output_sequence = pd.DataFrame([[0, 6], [16, 21]], columns=["start", "end"], dtype="Int64")

        window_length = 4
        test_output = self.algorithm_class(
            window_length_s=window_length,
            window_overlap=(window_length - 1) / window_length,
            window_overlap_samples=None,
            metric="maximum",
            inactive_signal_threshold=0,
        ).detect(test_input, sampling_rate_hz=1)
        assert_array_equal(test_output.per_sample_zupts_, expected_output)
        assert_frame_equal(test_output.zupts_, expected_output_sequence)
        assert test_output.min_vel_value_ == 0.0
        assert test_output.min_vel_index_ == 2

    def test_max_overlap_metric_max_w3(self) -> None:
        """Test binary input data on max metric with window size 3."""
        test_input = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
        test_input = pd.DataFrame(np.column_stack([test_input, test_input, test_input]), columns=SF_GYR)
        expected_output = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0])
        expected_output_sequence = pd.DataFrame([[0, 6], [10, 13], [16, 21]], columns=["start", "end"], dtype="Int64")

        window_length = 3
        test_output = self.algorithm_class(
            window_length_s=window_length,
            window_overlap=(window_length - 1) / window_length,
            window_overlap_samples=None,
            metric="maximum",
            inactive_signal_threshold=0,
        ).detect(test_input, sampling_rate_hz=1)
        assert_array_equal(test_output.per_sample_zupts_, expected_output)
        assert_frame_equal(test_output.zupts_, expected_output_sequence)
        assert test_output.min_vel_value_ == 0.0
        assert test_output.min_vel_index_ == 1

    def test_max_overlap_metric_max_w6_default_overlap(self) -> None:
        """Test binary input data on max metric with window size 6."""
        test_input = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
        test_input = pd.DataFrame(np.column_stack([test_input, test_input, test_input]), columns=SF_GYR)
        expected_output = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        expected_output_sequence = pd.DataFrame([[0, 6]], columns=["start", "end"], dtype="Int64")

        window_length = 6
        test_output = self.algorithm_class(
            window_length_s=window_length,
            window_overlap=(window_length - 1) / window_length,
            window_overlap_samples=None,
            metric="maximum",
            inactive_signal_threshold=0,
        ).detect(test_input, sampling_rate_hz=1)
        assert_array_equal(test_output.per_sample_zupts_, expected_output)
        assert_frame_equal(test_output.zupts_, expected_output_sequence)
        assert test_output.min_vel_value_ == 0.0
        assert test_output.min_vel_index_ == 3

    def test_max_overlap_max_w3_with_noise_default_overlap(self) -> None:
        """Test binary input data on mean metric with window size 4 after adding a bit of noise."""
        test_input = np.array([0, 0.1, 0, 0, 0.1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0.1, 0, 0, 1, 1])
        test_input = pd.DataFrame(np.column_stack([test_input, test_input, test_input]), columns=SF_GYR)
        expected_output = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], dtype=bool)

        window_length = 3
        test_output = self.algorithm_class(
            window_length_s=window_length,
            window_overlap=(window_length - 1) / window_length,
            window_overlap_samples=None,
            metric="mean",
            inactive_signal_threshold=0.1,
        ).detect(test_input, sampling_rate_hz=1)
        assert_array_equal(test_output.per_sample_zupts_, expected_output)
        assert test_output.min_vel_value_ == 0.0
        assert test_output.min_vel_index_ == 11

    def test_max_overlap_max_w3_with_noise(self) -> None:
        """Test binary input data on max metric with window size 4 after adding a bit of noise."""
        test_input = np.array([0, 0.1, 0, 0, 0.1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0.1, 0, 0, 1, 1])
        test_input = pd.DataFrame(np.column_stack([test_input, test_input, test_input]), columns=SF_GYR)
        expected_output = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        window_length = 3
        test_output = self.algorithm_class(
            window_length_s=window_length,
            window_overlap=(window_length - 1) / window_length,
            window_overlap_samples=None,
            metric="maximum",
            inactive_signal_threshold=0.1,
        ).detect(test_input, sampling_rate_hz=1)
        assert_array_equal(test_output.per_sample_zupts_, expected_output)
        assert test_output.min_vel_value_ == 0.0
        assert test_output.min_vel_index_ == 11

    def test_no_zupt_detected(self) -> None:
        test_input = np.array(
            [0.2, 0.4, 0.2, 0.2, 0.4, 0.2, 1, 1, 1, 1, 0.2, 0.2, 0.2, 1, 1, 1, 0.2, 0.2, 0.4, 0.2, 0.2, 1, 1]
        )
        test_input = pd.DataFrame(np.column_stack([test_input, test_input, test_input]), columns=SF_GYR)
        expected_output = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        window_length = 3
        test_output = self.algorithm_class(
            window_length_s=window_length,
            window_overlap=(window_length - 1) / window_length,
            window_overlap_samples=None,
            metric="maximum",
            inactive_signal_threshold=0.1,
        ).detect(test_input, sampling_rate_hz=1)
        assert_array_equal(test_output.per_sample_zupts_, expected_output)
        assert np.isnan(test_output.min_vel_value_)
        assert test_output.min_vel_index_ == 11

    def test_real_data_regression(self, healthy_example_imu_data, snapshot) -> None:
        """Test real data with default parameters."""
        test_output = self.algorithm_class().detect(healthy_example_imu_data["left_sensor"], sampling_rate_hz=204.8)
        snapshot.assert_match(test_output.zupts_)


class TestShoeZuptDetector:
    # Note: We don't retest all the validation here, as this is basically identical to the other ZUPT detectors.

    def test_real_data_regression(self, healthy_example_imu_data, snapshot) -> None:
        """Test real data with default parameters."""
        test_output = ShoeZuptDetector().detect(healthy_example_imu_data["left_sensor"], sampling_rate_hz=204.8)
        snapshot.assert_match(test_output.zupts_)

    def test_no_zupt_detected(self) -> None:
        test_input = np.array(
            [0.2, 0.4, 0.2, 0.2, 0.4, 0.2, 1, 1, 1, 1, 0.2, 0.2, 0.2, 1, 1, 1, 0.2, 0.2, 0.4, 0.2, 0.2, 1, 1]
        )
        test_input_with_gravity = np.add.outer(test_input, [0, 0, 9.81])
        test_input = pd.DataFrame(
            np.column_stack([test_input_with_gravity, test_input, test_input, test_input]), columns=SF_COLS
        )
        expected_output = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        window_length = 3
        test_output = ShoeZuptDetector(
            window_length_s=window_length,
            window_overlap=(window_length - 1) / window_length,
            window_overlap_samples=None,
            inactive_signal_threshold=0.1,
            acc_noise_variance=1,
            gyr_noise_variance=1,
        ).detect(test_input, sampling_rate_hz=1)
        assert_array_equal(test_output.per_sample_zupts_, expected_output)
        assert np.isnan(test_output.min_vel_value_)
        assert test_output.min_vel_index_ == 11
