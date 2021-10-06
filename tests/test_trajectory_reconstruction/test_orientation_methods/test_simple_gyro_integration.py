import pytest

from gaitmap.base import BaseOrientationMethod, BaseType
from gaitmap.trajectory_reconstruction import SimpleGyroIntegration
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin
from tests.mixins.test_caching_mixin import TestCachingMixin
from tests.test_trajectory_reconstruction.test_orientation_methods.test_ori_method_mixin import (
    TestOrientationMethodMixin,
)


class MetaTestConfig:
    algorithm_class = SimpleGyroIntegration

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data, healthy_example_stride_events) -> BaseType:
        position = SimpleGyroIntegration()
        position.estimate(healthy_example_imu_data["left_sensor"].iloc[:10], sampling_rate_hz=1)
        return position


class TestMetaFunctionality(MetaTestConfig, TestAlgorithmMixin):
    __test__ = True


class TestCachingFunctionality(MetaTestConfig, TestCachingMixin):
    __test__ = True


class TestSimpleRotations(TestOrientationMethodMixin):
    __test__ = True

    def init_algo_class(self) -> BaseOrientationMethod:
        return SimpleGyroIntegration()
