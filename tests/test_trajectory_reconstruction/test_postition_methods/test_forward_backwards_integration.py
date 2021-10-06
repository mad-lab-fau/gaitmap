import pytest

from gaitmap.base import BasePositionMethod, BaseType
from gaitmap.trajectory_reconstruction import ForwardBackwardIntegration
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin
from tests.test_trajectory_reconstruction.test_postition_methods.test_pos_method_mixin import (
    TestPositionMethodNoGravityMixin,
)


class MetaTestConfig:
    algorithm_class = ForwardBackwardIntegration

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data, healthy_example_stride_events) -> BaseType:
        position = ForwardBackwardIntegration()
        position.estimate(healthy_example_imu_data["left_sensor"].iloc[:10], sampling_rate_hz=1)
        return position


class TestMetaFunctionality(MetaTestConfig, TestAlgorithmMixin):
    __test__ = True


class TestSimpleIntegrationsNoGravity(TestPositionMethodNoGravityMixin):
    __test__ = True

    def init_algo_class(self) -> BasePositionMethod:
        # For basic integration tests, we do not remove gravity
        return ForwardBackwardIntegration(gravity=None)
