import pytest

from gaitmap.data_transform import (
    Resample,
    SlidingWindowMean,
    SlidingWindowStd,
    SlidingWindowVar,
    SlidingWindowGradient,
    BaseTransformer,
)
from gaitmap.utils.consts import BF_COLS
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin

all_rolling_transformer = [
    (SlidingWindowMean, {"window_size_s": 1}),
    (SlidingWindowStd, {"window_size_s": 1}),
    (SlidingWindowVar, {"window_size_s": 1}),
    (SlidingWindowGradient, {"window_size_s": 1}),
]


class TestMetaFunctionalityResample(TestAlgorithmMixin):
    __test__ = True

    algorithm_class = Resample

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data, healthy_example_stride_borders) -> Resample:
        data_left = healthy_example_imu_data["left_sensor"].iloc[:1000]
        data_left.columns = BF_COLS
        instance = self.algorithm_class(target_sampling_rate_hz=10)
        after_instance = instance.transform(
            data_left, roi_list=healthy_example_stride_borders["left_sensor"], sampling_rate_hz=100
        )
        return after_instance


class TestMetaFunctionalityRollingTransforms(TestAlgorithmMixin):
    __test__ = True

    @pytest.fixture(params=all_rolling_transformer, autouse=True)
    def set_algo_class(self, request):
        self.algorithm_class, self.algo_params = request.param

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data, healthy_example_stride_borders) -> BaseTransformer:
        data_left = healthy_example_imu_data["left_sensor"].iloc[:1000]
        data_left.columns = BF_COLS
        instance = self.algorithm_class(**self.algo_params)
        after_instance = instance.transform(data_left, sampling_rate_hz=100)
        return after_instance
