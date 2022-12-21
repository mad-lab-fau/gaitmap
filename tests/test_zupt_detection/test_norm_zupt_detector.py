import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from gaitmap.utils.consts import BF_ACC, BF_GYR, SF_ACC, SF_GYR
from gaitmap.utils.exceptions import ValidationError
from gaitmap.zupt_detection import NormZuptDetector, ShoeZuptDetector
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


class TestNormZuptDetector:
    """Test the function `sliding_window_view`."""

    @pytest.mark.parametrize("ws,sr", ((1, 1), (1, 2), (2, 1), (2.49, 1)))
    def test_error_window_to_small(self, healthy_example_imu_data, ws, sr):
        with pytest.raises(ValidationError, match=r".*The effective window size is smaller*"):
            NormZuptDetector(window_length_s=ws).detect(healthy_example_imu_data["left_sensor"], sr)

    @pytest.mark.parametrize("overlap,valid", ((0, True), (1, False), (0.5, True), (-0.5, False)))
    def test_wrong_window_overlap(self, overlap, valid, healthy_example_imu_data):
        if not valid:
            with pytest.raises(ValidationError, match=r"`window_overlap` must be `0 <= window_overlap < 1`"):
                NormZuptDetector(window_overlap=overlap).detect(healthy_example_imu_data["left_sensor"], 200)
        else:
            _ = NormZuptDetector(window_overlap=overlap).detect(healthy_example_imu_data["left_sensor"], 200)

    @pytest.mark.parametrize("win_len,overlap,valid", ((1, 0, True), (10 / 200, 0.99, False)))
    def test_effective_overlap_error(self, win_len, overlap, valid, healthy_example_imu_data):
        if not valid:
            with pytest.raises(ValidationError, match=r".*The effective window overlap after rounding is 1"):
                NormZuptDetector(window_overlap=overlap, window_length_s=win_len).detect(
                    healthy_example_imu_data["left_sensor"], 200
                )
        else:
            NormZuptDetector(window_overlap=overlap, window_length_s=win_len).detect(
                healthy_example_imu_data["left_sensor"], 200
            )

    @pytest.mark.parametrize(
        "sensor,axis,valid",
        (
            ("acc", SF_ACC, True),
            ("acc", BF_ACC, True),
            ("acc", BF_GYR, False),
            ("gyr", BF_GYR, True),
            ("gyr", SF_GYR, True),
            ("gyr", SF_ACC, False),
        ),
    )
    def test_single_sensor_data(self, sensor, axis, valid):
        data = pd.DataFrame(np.empty((10, len(axis))), columns=axis)
        if valid is False:
            with pytest.raises(ValidationError):
                NormZuptDetector(sensor=sensor, window_length_s=0.5).detect(data, 10)
        else:
            NormZuptDetector(sensor=sensor, window_length_s=0.5).detect(data, 10)

    def test_debug_outputs(self):
        data = pd.DataFrame(np.empty((10, 3)), columns=BF_GYR)
        zupt = NormZuptDetector(sensor="gyr", window_length_s=0.5, window_overlap=0.2).detect(data, 10)
        assert zupt.window_length_samples_ == 5
        assert zupt.window_overlap_samples_ == 1

    def test_invalid_input_metric_default_overlap(self, healthy_example_imu_data):
        """Test if value error is raised correctly on invalid input dimensions."""
        with pytest.raises(ValueError, match=r".*Invalid metric passed!.*"):
            NormZuptDetector(metric="invalid").detect(healthy_example_imu_data["left_sensor"], 204.8)

    def test_invalid_window_length(self, healthy_example_imu_data):
        """Test if value error is raised when window longer than signal."""
        with pytest.raises(ValueError, match=r".*Invalid window length*"):
            NormZuptDetector(window_length_s=1000).detect(healthy_example_imu_data["left_sensor"].iloc[:500], 1.0)

    def test_single_window_fit(self):
        """Test input where only a single window length fits within input signal."""
        test_input = np.array([0, 0, 0, 0, 0, 1, 1, 1])
        test_input = pd.DataFrame(np.column_stack([test_input, test_input, test_input]), columns=SF_GYR)
        expected_output = np.array([1, 1, 1, 1, 1, 0, 0, 0])
        expected_output_sequence = pd.DataFrame([[0, 5]], columns=["start", "end"])
        window_length = 5
        test_output = NormZuptDetector(window_length_s=window_length, window_overlap=0).detect(
            test_input, sampling_rate_hz=1
        )
        assert_array_equal(test_output.per_sample_zupts_, expected_output)
        assert_frame_equal(test_output.zupts_, expected_output_sequence)

    def test_max_overlap_metric_max_w4_default_overlap(self):
        """Test binary input data on max metric with window size 4."""
        test_input = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
        test_input = pd.DataFrame(np.column_stack([test_input, test_input, test_input]), columns=SF_GYR)
        expected_output = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
        expected_output_sequence = pd.DataFrame([[0, 6], [16, 21]], columns=["start", "end"])

        window_length = 4
        test_output = NormZuptDetector(
            window_length_s=window_length,
            window_overlap=(window_length - 1) / window_length,
            metric="maximum",
            inactive_signal_threshold=0,
        ).detect(test_input, sampling_rate_hz=1)
        assert_array_equal(test_output.per_sample_zupts_, expected_output)
        assert_frame_equal(test_output.zupts_, expected_output_sequence)

    def test_max_overlap_metric_max_w3(self):
        """Test binary input data on max metric with window size 3."""
        test_input = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
        test_input = pd.DataFrame(np.column_stack([test_input, test_input, test_input]), columns=SF_GYR)
        expected_output = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0])
        expected_output_sequence = pd.DataFrame([[0, 6], [10, 13], [16, 21]], columns=["start", "end"])

        window_length = 3
        test_output = NormZuptDetector(
            window_length_s=window_length,
            window_overlap=(window_length - 1) / window_length,
            metric="maximum",
            inactive_signal_threshold=0,
        ).detect(test_input, sampling_rate_hz=1)
        assert_array_equal(test_output.per_sample_zupts_, expected_output)
        assert_frame_equal(test_output.zupts_, expected_output_sequence)

    def test_max_overlap_metric_max_w6_default_overlap(self):
        """Test binary input data on max metric with window size 6."""
        test_input = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
        test_input = pd.DataFrame(np.column_stack([test_input, test_input, test_input]), columns=SF_GYR)
        expected_output = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        expected_output_sequence = pd.DataFrame([[0, 6]], columns=["start", "end"])

        window_length = 6
        test_output = NormZuptDetector(
            window_length_s=window_length,
            window_overlap=(window_length - 1) / window_length,
            metric="maximum",
            inactive_signal_threshold=0,
        ).detect(test_input, sampling_rate_hz=1)
        assert_array_equal(test_output.per_sample_zupts_, expected_output)
        assert_frame_equal(test_output.zupts_, expected_output_sequence)

    def test_max_overlap_max_w3_with_noise_default_overlap(self):
        """Test binary input data on mean metric with window size 4 after adding a bit of noise."""
        test_input = np.array([0, 0.1, 0, 0, 0.1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0.1, 0, 0, 1, 1])
        test_input = pd.DataFrame(np.column_stack([test_input, test_input, test_input]), columns=SF_GYR)
        expected_output = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], dtype=bool)

        window_length = 3
        test_output = NormZuptDetector(
            window_length_s=window_length,
            window_overlap=(window_length - 1) / window_length,
            metric="mean",
            inactive_signal_threshold=0.1,
        ).detect(test_input, sampling_rate_hz=1)
        assert_array_equal(test_output.per_sample_zupts_, expected_output)

    def test_max_overlap_max_w3_with_noise(self):
        """Test binary input data on max metric with window size 4 after adding a bit of noise."""
        test_input = np.array([0, 0.1, 0, 0, 0.1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0.1, 0, 0, 1, 1])
        test_input = pd.DataFrame(np.column_stack([test_input, test_input, test_input]), columns=SF_GYR)
        expected_output = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        window_length = 3
        test_output = NormZuptDetector(
            window_length_s=window_length,
            window_overlap=(window_length - 1) / window_length,
            metric="maximum",
            inactive_signal_threshold=0.1,
        ).detect(test_input, sampling_rate_hz=1)
        assert_array_equal(test_output.per_sample_zupts_, expected_output)


class TestShoeZuptDetector:
    @pytest.mark.parametrize("ws,sr", ((1, 1), (1, 2), (2, 1), (2.49, 1)))
    def test_error_window_to_small(self, healthy_example_imu_data, ws, sr):
        with pytest.raises(ValidationError, match=r".*The effective window size is smaller*"):
            ShoeZuptDetector(window_length_s=ws).detect(healthy_example_imu_data["left_sensor"], sr)

    @pytest.mark.parametrize("overlap,valid", ((0, True), (1, False), (0.5, True), (-0.5, False)))
    def test_wrong_window_overlap(self, overlap, valid, healthy_example_imu_data):
        if not valid:
            with pytest.raises(ValidationError, match=r"`window_overlap` must be `0 <= window_overlap < 1`"):
                ShoeZuptDetector(window_overlap=overlap).detect(healthy_example_imu_data["left_sensor"], 200)
        else:
            _ = ShoeZuptDetector(window_overlap=overlap).detect(healthy_example_imu_data["left_sensor"], 200)

    @pytest.mark.parametrize("win_len,overlap,valid", ((1, 0, True), (10 / 200, 0.99, False)))
    def test_effective_overlap_error(self, win_len, overlap, valid, healthy_example_imu_data):
        if not valid:
            with pytest.raises(ValidationError, match=r".*The effective window overlap after rounding is 1"):
                ShoeZuptDetector(window_overlap=overlap, window_length_s=win_len).detect(
                    healthy_example_imu_data["left_sensor"], 200
                )
        else:
            ShoeZuptDetector(window_overlap=overlap, window_length_s=win_len).detect(
                healthy_example_imu_data["left_sensor"], 200
            )