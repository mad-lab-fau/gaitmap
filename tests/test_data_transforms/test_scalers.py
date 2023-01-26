import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from gaitmap.base import BaseType
from gaitmap.data_transform import (
    AbsMaxScaler,
    FixedScaler,
    MinMaxScaler,
    StandardScaler,
    TrainableAbsMaxScaler,
    TrainableMinMaxScaler,
    TrainableStandardScaler,
    TrainableTransformerMixin,
)
from gaitmap.utils.consts import BF_COLS
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin

all_scaler = [
    TrainableMinMaxScaler,
    MinMaxScaler,
    FixedScaler,
    AbsMaxScaler,
    TrainableAbsMaxScaler,
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


class TestFixedScaler:
    def test_transform_default(self):
        t = FixedScaler()
        data = pd.DataFrame(np.random.rand(10, 10))
        t.transform(data)
        assert id(t.data) == id(data)
        # By default the transformer should not change the data
        assert t.transformed_data_.equals(data)

    @pytest.mark.parametrize(("scale", "offset"), [(1, 1), (2, 1), (3, 2)])
    def test_transform(self, scale, offset):
        t = FixedScaler(scale=scale, offset=offset)
        data = pd.DataFrame(np.random.rand(10, 10))
        t.transform(data)
        assert id(t.data) == id(data)
        assert t.transformed_data_.equals((data - offset) / scale)


class TestStandardScaler:
    @pytest.mark.parametrize("ddof", [0, 1, 2])
    def test_transform(self, ddof):
        t = StandardScaler(ddof=ddof)
        data = pd.DataFrame(np.random.rand(10, 10))
        t.transform(data)
        assert id(t.data) == id(data)
        assert t.transformed_data_.to_numpy().std(ddof=ddof) == pytest.approx(1, rel=1e-3)
        assert t.transformed_data_.to_numpy().mean() == pytest.approx(0, rel=1e-3)


class TestTrainableStandardScaler:
    @pytest.mark.parametrize("ddof", [0, 1, 2])
    def test_transform(self, ddof):
        t = TrainableStandardScaler(ddof=ddof)
        train_data = pd.DataFrame(np.random.rand(10, 10))
        test_data = pd.DataFrame(np.random.rand(10, 10))
        t.self_optimize([train_data])

        # Train results
        assert t.std == train_data.to_numpy().std(ddof=ddof)
        assert t.mean == train_data.to_numpy().mean()

        # On train data
        t.transform(train_data)
        assert id(t.data) == id(train_data)
        assert t.transformed_data_.to_numpy().std(ddof=ddof) == pytest.approx(1, rel=1e-3)
        assert t.transformed_data_.to_numpy().mean() == pytest.approx(0, rel=1e-3)

        # On test data
        t.transform(test_data)
        assert id(t.data) == id(test_data)
        assert t.transformed_data_.equals(
            (test_data - train_data.to_numpy().mean()) / train_data.to_numpy().std(ddof=ddof)
        )

    def test_iterative_std_calculation(self):
        test_data = [pd.DataFrame(np.random.rand(10, 10)) for _ in range(5)]
        test_data_concatenated = pd.concat(test_data)
        t = TrainableStandardScaler()

        t.self_optimize(test_data)

        assert t.std == pytest.approx(test_data_concatenated.to_numpy().std(ddof=t.ddof), rel=1e-5)

    def test_raise_error_before_optimization(self):
        t = TrainableStandardScaler()
        with pytest.raises(ValueError):
            t.transform(pd.DataFrame(np.random.rand(10, 10)))


class TestAbsMaxScaler:
    @pytest.mark.parametrize("out_max", [2, 3, 0.3])
    @pytest.mark.parametrize("data_factor", [1, -2, 0.3])
    def test_transform(self, out_max, data_factor):
        t = AbsMaxScaler(out_max=out_max)
        data = pd.DataFrame([[0, 1, 1], [2, 1, 3]]) * data_factor
        t.transform(data)
        assert id(t.data) == id(data)
        assert t.transformed_data_.abs().to_numpy().max() == pytest.approx(out_max, rel=1e-3)
        assert_frame_equal(t.transformed_data_, data / np.max(np.abs(data.to_numpy())) * out_max)


class TestTrainableAbsMaxScaler:
    @pytest.mark.parametrize("out_max", [2, 3, 0.3])
    def test_transform(self, out_max):
        t = TrainableAbsMaxScaler(out_max=out_max)
        train_data = pd.DataFrame(np.random.rand(10, 10))
        test_data = pd.DataFrame(np.random.rand(10, 10))
        t.self_optimize([train_data])

        # Train results
        assert t.data_max == np.max(np.abs(train_data.to_numpy()))

        # On train data
        t.transform(train_data)
        assert id(t.data) == id(train_data)
        assert t.transformed_data_.abs().to_numpy().max() == pytest.approx(out_max, rel=1e-3)

        # On test data
        t.transform(test_data)
        assert id(t.data) == id(test_data)
        assert_frame_equal(t.transformed_data_, test_data / np.max(np.abs(train_data.to_numpy())) * out_max)

    def test_raise_error_before_optimization(self):
        t = TrainableAbsMaxScaler()
        with pytest.raises(ValueError):
            t.transform(pd.DataFrame(np.random.rand(10, 10)))


class TestMinMaxScaler:
    @pytest.mark.parametrize("out_range", [(0, 1), (0, 2), (-1, 1), (-1, 2)])
    def test_transform(self, out_range):
        t = MinMaxScaler(out_range=out_range)
        data = pd.DataFrame(np.random.rand(10, 10))
        t.transform(data)
        assert id(t.data) == id(data)
        assert t.transformed_data_.to_numpy().min() == pytest.approx(out_range[0], rel=1e-3)
        assert t.transformed_data_.to_numpy().max() == pytest.approx(out_range[1], rel=1e-3)

    @pytest.mark.parametrize("out_range", [(0, 0), (1, -1), (2, 2)])
    def test_raise_error_for_invalid_out_range(self, out_range):
        data = pd.DataFrame(np.random.rand(10, 10))
        t = MinMaxScaler(out_range=out_range)
        with pytest.raises(ValueError):
            t.transform(data)


class TestTrainableMinMaxScaler:
    @pytest.mark.parametrize("out_range", [(0, 1), (0, 2), (-1, 1), (-1, 2)])
    def test_transform(self, out_range):
        t = TrainableMinMaxScaler(out_range=out_range)
        train_data = pd.DataFrame(np.random.rand(10, 10))
        test_data = pd.DataFrame(np.random.rand(10, 10))
        t.self_optimize([train_data])

        # Train results
        assert t.data_range == (train_data.to_numpy().min(), train_data.to_numpy().max())

        # On train data
        t.transform(train_data)
        assert id(t.data) == id(train_data)
        assert t.transformed_data_.to_numpy().min() == pytest.approx(out_range[0], rel=1e-3)
        assert t.transformed_data_.to_numpy().max() == pytest.approx(out_range[1], rel=1e-3)

        # On test data
        t.transform(test_data)
        assert id(t.data) == id(test_data)
        assert_frame_equal(
            t.transformed_data_,
            (test_data - train_data.to_numpy().min())
            / (train_data.to_numpy().max() - train_data.to_numpy().min())
            * (out_range[1] - out_range[0])
            + out_range[0],
        )

    def test_raise_error_before_optimization(self):
        t = TrainableMinMaxScaler()
        with pytest.raises(ValueError):
            t.transform(pd.DataFrame(np.random.rand(10, 10)))
