from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

from gaitmap.base import BaseType, BaseZuptDetector
from gaitmap.trajectory_reconstruction import RtsKalman
from gaitmap.trajectory_reconstruction.trajectory_methods._rts_kalman import MadgwickRtsKalman
from gaitmap.utils.consts import SF_COLS
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin
from tests.test_trajectory_reconstruction.test_trajectory_methods.test_trajectory_method_mixin import (
    TestTrajectoryMethodMixin,
)


class TestMetaFunctionalityRtsKalman(TestAlgorithmMixin):
    __test__ = True
    algorithm_class = RtsKalman

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data, healthy_example_stride_events) -> BaseType:
        kalman_filter = RtsKalman()
        kalman_filter.estimate(healthy_example_imu_data["left_sensor"].iloc[:15], sampling_rate_hz=100)
        return kalman_filter


class TestMetaFunctionalityMadgwickRtsKalman(TestAlgorithmMixin):
    __test__ = True
    algorithm_class = MadgwickRtsKalman

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data, healthy_example_stride_events) -> BaseType:
        kalman_filter = MadgwickRtsKalman()
        kalman_filter.estimate(healthy_example_imu_data["left_sensor"].iloc[:15], sampling_rate_hz=100)
        return kalman_filter


class TestTrajectoryMethod(TestTrajectoryMethodMixin):
    __test__ = True

    default_kwargs = {}

    def init_algo_class(self, **kwargs) -> RtsKalman:
        kwargs = {**self.default_kwargs, **kwargs}
        return RtsKalman().set_params(**kwargs)

    def test_covariance_output_format(self) -> None:
        test = self.init_algo_class(zupt_detector__window_length_s=1)
        fs = 15
        sensor_data = np.repeat(np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])[None, :], fs, axis=0)
        sensor_data = pd.DataFrame(sensor_data, columns=SF_COLS)
        test.estimate(sensor_data, sampling_rate_hz=fs)

        assert test.covariance_.shape == (len(sensor_data) + 1, 9 * 9)

    def test_zupt_output(self) -> None:
        test = self.init_algo_class(
            zupt_detector__inactive_signal_threshold=10,
            zupt_detector__window_length_s=0.3,
            zupt_detector__window_overlap=0.8,
        )
        fs = 15
        sensor_data = np.repeat(np.array([0.0, 0.0, 9.81, 0.0, 0.0, 100.0])[None, :], 100, axis=0)
        expected_zupts = [[0, 10], [30, 55], [85, 90]]
        for z in expected_zupts:
            sensor_data[slice(*z), -1] = 0
        sensor_data = pd.DataFrame(sensor_data, columns=SF_COLS)
        test.estimate(sensor_data, sampling_rate_hz=fs)

        assert_array_almost_equal(expected_zupts, test.zupts_)

    def test_corrects_velocity_drift(self) -> None:
        """Check that ZUPTs correct a velocity drift and set velocity to zero."""
        test = self.init_algo_class(zupt_detector__window_length_s=0.3, level_walking=False)
        acc = np.array([5.0, 5.0, 12.81])
        accel_data = np.repeat(np.concatenate((acc, [0.0, 0.0, 40.0]))[None, :], 5, axis=0)
        zupt_data = np.repeat(np.concatenate((acc, [0.0, 0.0, 0.0]))[None, :], 10, axis=0)
        sensor_data = np.vstack((accel_data, zupt_data))
        sensor_data = pd.DataFrame(sensor_data, columns=SF_COLS)
        test.estimate(sensor_data, sampling_rate_hz=10)
        assert_array_almost_equal(test.velocity_.to_numpy()[-1], [0.0, 0.0, 0.0], decimal=10)

    def test_corrects_z_position(self) -> None:
        """Check that level walking reset position to zero during ZUPTs."""
        test = self.init_algo_class(zupt_detector__window_length_s=1)
        accel_data = np.repeat(np.concatenate(([0.0, 0.0, 100], [0.0, 0.0, 40.0]))[None, :], 5, axis=0)
        zupt_data = np.repeat(np.concatenate(([0.0, 0.0, 9.81], [0.0, 0.0, 0.0]))[None, :], 10, axis=0)
        sensor_data = np.vstack((accel_data, zupt_data))
        sensor_data = pd.DataFrame(sensor_data, columns=SF_COLS)
        test.estimate(sensor_data, sampling_rate_hz=10)
        assert test.position_.to_numpy()[4][2] < -0.8
        assert_array_almost_equal(test.position_.to_numpy()[-1], [0.0, 0.0, 0.0], decimal=10)

    def test_stride_list_forwarded_to_zupt(self) -> None:
        """Test that the stride list passed to reconstruct is forwarded to the detect method of the ZUPT detector."""

        class MockZUPTDetector(BaseZuptDetector):
            zupts_ = pd.DataFrame(columns=["start", "end"])
            per_sample_zupts_ = np.zeros(10)

        with patch.object(MockZUPTDetector, "detect") as mock_detect:
            mock_detect.return_value = MockZUPTDetector()
            test = self.init_algo_class(zupt_detector=MockZUPTDetector())
            sensor_data = pd.DataFrame(np.zeros((10, 6)), columns=SF_COLS)
            stride_event_list = pd.DataFrame({"start": [0, 5], "end": [2, 7]}, index=pd.Series([0, 1], name="s_id"))
            test.estimate(sensor_data, sampling_rate_hz=10, stride_event_list=stride_event_list)

            mock_detect.assert_called_once_with(sensor_data, sampling_rate_hz=10, stride_event_list=stride_event_list)


class TestMadgwickKalman(TestTrajectoryMethod):
    """Test the Madgwick Kalman.

    For beta = 0 this should be identical to normal Kalman
    """

    def init_algo_class(self, **kwargs) -> RtsKalman:
        kwargs = {**self.default_kwargs, **kwargs}
        return MadgwickRtsKalman().set_params(**kwargs)
