import numpy as np
import pandas as pd
import pytest

from gaitmap.base import BaseType, BaseOrientationMethod
from gaitmap.trajectory_reconstruction import RtsKalman
from gaitmap.utils.consts import SF_COLS
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin
from tests.test_trajectory_reconstruction.test_trajectory_method_mixin import TestTrajectoryMethodMixin


class TestMetaFunctionality(TestAlgorithmMixin):
    algorithm_class = RtsKalman
    __test__ = True

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data, healthy_example_stride_events) -> BaseType:
        kalman_filter = RtsKalman()
        kalman_filter.estimate(
            healthy_example_imu_data["left_sensor"].iloc[:15], sampling_rate_hz=1,
        )
        return kalman_filter


class TestTrajectoryMethod(TestTrajectoryMethodMixin):
    __test__ = True

    def init_algo_class(self) -> BaseOrientationMethod:
        return RtsKalman()

    def test_covariance_output_format(self):
        test = self.init_algo_class()
        fs = 15
        sensor_data = np.repeat(np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])[None, :], fs, axis=0)
        sensor_data = pd.DataFrame(sensor_data, columns=SF_COLS)
        test.estimate(sensor_data, 1.0 / fs)

        assert len(test.covariance_) == (len(sensor_data) + 1) * 9
        assert len(test.covariance_.columns) == 9
        assert test.covariance_.index.levshape == (len(sensor_data) + 1, 9)
