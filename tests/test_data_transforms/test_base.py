import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from gaitmap.base import BaseType
from gaitmap.data_transform import (
    IdentityTransformer,
    TrainableMinMaxScaler,
    GroupedTransformer,
    TrainableAbsMaxScaler,
    TrainableTransformerMixin, BaseTransformer, FixedScaler,
)
from gaitmap.utils.consts import BF_COLS
from mixins.test_algorithm_mixin import TestAlgorithmMixin

all_base_transformer = [
    TrainableMinMaxScaler,
    GroupedTransformer,
    IdentityTransformer,
]


class TestMetaFunctionality(TestAlgorithmMixin):
    __test__ = True

    @pytest.fixture(params=all_base_transformer, autouse=True)
    def set_algo_class(self, request):
        self.algorithm_class = request.param

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data, healthy_example_stride_borders) -> BaseTransformer:
        data_left = healthy_example_imu_data["left_sensor"].iloc[:10]
        data_left.columns = BF_COLS
        instance = self.algorithm_class()
        if isinstance(instance, TrainableTransformerMixin):
            instance = instance.self_optimize([data_left])
        after_instance = instance.transform(data_left)
        return after_instance


class TestIdentityTransformer:
    def test_transform(self):
        t = IdentityTransformer()
        data = pd.DataFrame(np.random.rand(10, 3))
        t.transform(data)

        assert_frame_equal(t.transformed_data_, data)
        assert id(t.data) == id(data)
        assert id(t.transformed_data_) != id(data)


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


class TestGroupedTransformer:
    @pytest.mark.parametrize("keep_all_cols", [True, False])
    def test_transform_no_opti(self, keep_all_cols):
        data = pd.DataFrame(np.ones((10, 3)), columns=list("abc"))
        t = GroupedTransformer(
            transformer_mapping=[("b", FixedScaler(3)), ("a", FixedScaler(2))], keep_all_cols=keep_all_cols
        )
        t.transform(data)

        assert id(t.data) == id(data)
        assert_frame_equal(t.transformed_data_[["a"]], data[["a"]] / 2)
        assert_frame_equal(t.transformed_data_[["b"]], data[["b"]] / 3)
        if keep_all_cols:
            assert_frame_equal(t.transformed_data_[["c"]], data[["c"]])
        else:
            assert "c" not in t.transformed_data_.columns

        # Test that the order of columns matches the data
        assert t.transformed_data_.columns.tolist() == ["a", "b", "c"] if keep_all_cols else ["a", "b"]

    def test_multi_scale(self):
        data = pd.DataFrame(np.ones((10, 3)), columns=list("abc"))
        t = GroupedTransformer(transformer_mapping=[(("a", "b", "c"), FixedScaler(3))])
        t.transform(data)

        assert id(t.data) == id(data)
        assert_frame_equal(t.transformed_data_, data / 3.0)

    def test_error_when_transformer_not_unique(self):
        scaler = FixedScaler(3)
        t = GroupedTransformer(transformer_mapping=[("b", scaler), ("a", scaler)])
        with pytest.raises(ValueError):
            t.self_optimize([pd.DataFrame(np.ones((10, 3)))])

    def test_error_attempting_double_transform(self):
        t = GroupedTransformer(transformer_mapping=[("b", FixedScaler()), ("b", FixedScaler())])
        with pytest.raises(ValueError):
            t.transform(pd.DataFrame(np.ones((10, 3))))

    def test_optimization(self):
        data = pd.DataFrame(np.ones((10, 3)), columns=list("abc"))
        scale_vals = [1, 2, 3]
        train_data = pd.DataFrame(np.ones((10, 3)), columns=list("abc")) * scale_vals
        t = GroupedTransformer(
            [("a", TrainableAbsMaxScaler()), ("b", TrainableAbsMaxScaler()), ("c", TrainableAbsMaxScaler())]
        )
        t.self_optimize([train_data])

        # Train results
        for (_, tr), val in zip(t.transformer_mapping, scale_vals):
            assert tr.data_max == val

        t.transform(data)

        assert_frame_equal(t.transformed_data_, data / np.array(scale_vals))

    def test_none_transformer(self):
        t = GroupedTransformer(None)
        data = pd.DataFrame(np.ones((10, 3)), columns=list("abc"))
        t.self_optimize([data])
        t.transform(data)

        assert id(t.data) == id(data)
        assert id(t.transformed_data_) != id(data)
        assert_frame_equal(t.transformed_data_, data)
