import numpy as np
import pytest

from gaitmap.base import BaseType, BaseOrientationMethod
from gaitmap.trajectory_reconstruction import MadgwickAHRS
from gaitmap.trajectory_reconstruction.orientation_methods.madgwick import _madgwick_update
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin
from tests.mixins.test_caching_mixin import TestCachingMixin
from tests.test_trajectory_reconstruction.test_orientation_methods.test_ori_method_mixin import (
    TestOrientationMethodMixin,
)


class MetaTestConfig:
    algorithm_class = MadgwickAHRS

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data, healthy_example_stride_events) -> BaseType:
        position = MadgwickAHRS()
        position.estimate(
            healthy_example_imu_data["left_sensor"].iloc[:10], sampling_rate_hz=1,
        )
        return position


class TestMetaFunctionality(MetaTestConfig, TestAlgorithmMixin):
    __test__ = True


class TestCachingFunctionality(MetaTestConfig, TestCachingMixin):
    __test__ = True


class TestSimpleRotations(TestOrientationMethodMixin):
    __test__ = True

    def init_algo_class(self) -> BaseOrientationMethod:
        return MadgwickAHRS()

    def test_correction_works(self):
        """Madgwick should be able to resist small roations if acc does not change."""
        ori = np.array([0, 0, 0, 1.0])
        initial_ori = ori
        for i in range(50):
            ori = _madgwick_update(
                np.array([1, 1, 0.0]), np.array([0, 0.0, 1.0]), initial_orientation=ori, sampling_rate_hz=50, beta=1.0,
            )

        np.testing.assert_array_almost_equal(ori, initial_ori, decimal=2)
