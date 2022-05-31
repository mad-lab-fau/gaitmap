import pytest

from gaitmap.base import BaseType
from gaitmap.data_transform import (
    BaseTransformer,
    TrainableMinMaxScaler,
    MinMaxScaler,
    GroupedTransformer,
    TrainableTransformerMixin,
    FixedScaler,
    AbsMaxScaler,
    TrainableAbsMaxScaler,
    IdentityTransformer,
    StandardScaler,
    TrainableStandardScaler,
)
from gaitmap.utils.consts import BF_COLS
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin

all_scaler = [
    TrainableMinMaxScaler,
    MinMaxScaler,
    GroupedTransformer,
    FixedScaler,
    AbsMaxScaler,
    TrainableAbsMaxScaler,
    IdentityTransformer,
    StandardScaler,
    TrainableStandardScaler,
]


class TestMetaFunctionality(TestAlgorithmMixin):
    __test__ = True

    @pytest.fixture(params=all_scaler, autouse=True)
    def set_algo_class(self, request):
        self.algorithm_class = request.param

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data, healthy_example_stride_borders) -> BaseType:
        data_left = healthy_example_imu_data["left_sensor"].iloc[:10]
        data_left.columns = BF_COLS
        instance = self.algorithm_class()
        if isinstance(instance, TrainableTransformerMixin):
            instance = instance.self_optimize([data_left])
        after_instance = instance.transform(data_left)
        return after_instance
