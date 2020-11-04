import numpy as np
import pytest

from gaitmap.base import BaseType, BaseOrientationMethod
from gaitmap.trajectory_reconstruction.rts_kalman import RtsKalman
from gaitmap.trajectory_reconstruction.orientation_methods.madgwick import _madgwick_update
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin
from tests.test_trajectory_reconstruction.test_orientation_methods.test_ori_method_mixin import (
    TestOrientationMethodMixin,
)
from tests.test_trajectory_reconstruction.test_postition_methods.test_pos_method_mixin import (
    TestPositionMethodNoGravityMixin,
)


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


class TestSimpleRotations(TestOrientationMethodMixin):
    __test__ = True

    def init_algo_class(self) -> BaseOrientationMethod:
        return RtsKalman()


"""
class TestSimpleAcceleration(TestPositionMethodNoGravityMixin):
    __test__ = True

    def init_algo_class(self) -> BaseOrientationMethod:
        return RtsKalman()
"""
