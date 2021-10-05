import numpy as np
import pandas as pd
import pytest
from numpy.linalg import norm
from numpy.testing import assert_array_almost_equal

from gaitmap.base import BaseType
from gaitmap.trajectory_reconstruction import RtsKalman
from gaitmap.utils.consts import SF_COLS
from gaitmap.utils.rotations import find_unsigned_3d_angle
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin
from tests.mixins.test_caching_mixin import TestCachingMixin
from tests.test_trajectory_reconstruction.test_trajectory_methods.test_trajectory_method_mixin import (
    TestTrajectoryMethodMixin,
)


class MetaTestConfig:
    algorithm_class = RtsKalman

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data, healthy_example_stride_events) -> BaseType:
        kalman_filter = RtsKalman()
        kalman_filter.estimate(healthy_example_imu_data["left_sensor"].iloc[:15], sampling_rate_hz=1)
        return kalman_filter


class TestMetaFunctionality(MetaTestConfig, TestAlgorithmMixin):
    __test__ = True


class TestTrajectoryMethod(TestTrajectoryMethodMixin):
    __test__ = True

    default_kwargs = {}

    def init_algo_class(self, **kwargs) -> RtsKalman:
        kwargs = {**self.default_kwargs, **kwargs}
        return RtsKalman(**kwargs)

    def test_covariance_output_format(self):
        test = self.init_algo_class()
        fs = 15
        sensor_data = np.repeat(np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])[None, :], fs, axis=0)
        sensor_data = pd.DataFrame(sensor_data, columns=SF_COLS)
        test.estimate(sensor_data, fs)

        assert test.covariance_.shape == (len(sensor_data) + 1, 9 * 9)

    def test_zupt_output(self):
        test = self.init_algo_class(zupt_threshold_dps=10, zupt_window_length_s=0.01)
        fs = 15
        sensor_data = np.repeat(np.array([0.0, 0.0, 9.81, 0.0, 0.0, 100.0])[None, :], 100, axis=0)
        expected_zupts = [[0, 10], [30, 55], [85, 90]]
        for z in expected_zupts:
            sensor_data[slice(*z), -1] = 0
        sensor_data = pd.DataFrame(sensor_data, columns=SF_COLS)
        test.estimate(sensor_data, fs)

        assert_array_almost_equal(expected_zupts, test.zupts_)

    def test_corrects_velocity_drift(self):
        """Check that ZUPTs correct a velocity drift and set velocity to zero."""
        test = RtsKalman(level_walking=False, zupt_window_length_s=0.2)
        acc = np.array([5.0, 5.0, 12.81])
        accel_data = np.repeat(np.concatenate((acc, [0.0, 0.0, 40.0]))[None, :], 5, axis=0)
        zupt_data = np.repeat(np.concatenate((acc, [0.0, 0.0, 0.0]))[None, :], 10, axis=0)
        sensor_data = np.vstack((accel_data, zupt_data))
        sensor_data = pd.DataFrame(sensor_data, columns=SF_COLS)
        test.estimate(sensor_data, 10)
        assert norm(test.velocity_) > 1.0
        assert_array_almost_equal(test.velocity_.to_numpy()[-1], [0.0, 0.0, 0.0], decimal=10)

    def test_corrects_z_position(self):
        """Check that level walking reset position to zero during ZUPTs."""
        test = self.init_algo_class()
        accel_data = np.repeat(np.concatenate(([0.0, 0.0, 100], [0.0, 0.0, 40.0]))[None, :], 5, axis=0)
        zupt_data = np.repeat(np.concatenate(([0.0, 0.0, 9.81], [0.0, 0.0, 0.0]))[None, :], 10, axis=0)
        sensor_data = np.vstack((accel_data, zupt_data))
        sensor_data = pd.DataFrame(sensor_data, columns=SF_COLS)
        test.estimate(sensor_data, 10)
        assert test.position_.to_numpy()[4][2] < -0.8
        assert_array_almost_equal(test.position_.to_numpy()[-1], [0.0, 0.0, 0.0], decimal=10)
