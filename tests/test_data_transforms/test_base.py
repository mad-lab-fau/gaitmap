import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from tpcp import make_optimize_safe, make_action_safe

from gaitmap.data_transform import (
    BaseTransformer,
    ChainedTransformer,
    FixedScaler,
    GroupedTransformer,
    IdentityTransformer,
    ParallelTransformer,
    TrainableAbsMaxScaler,
    TrainableTransformerMixin,
)
from gaitmap.utils.consts import BF_COLS
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin

all_base_transformer = [
    GroupedTransformer,
    ChainedTransformer,
    ParallelTransformer,
    IdentityTransformer,
]


class TestMetaFunctionality(TestAlgorithmMixin):
    __test__ = True

    @pytest.fixture(params=all_base_transformer, autouse=True)
    def set_algo_class(self, request):
        self.algorithm_class = request.param

    def test_empty_init(self):
        pytest.skip()

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data, healthy_example_stride_borders) -> BaseTransformer:
        data_left = healthy_example_imu_data["left_sensor"].iloc[:10]
        data_left.columns = BF_COLS
        if self.algorithm_class is GroupedTransformer:
            instance = self.algorithm_class([(tuple(BF_COLS), IdentityTransformer())])
        elif self.algorithm_class is ChainedTransformer:
            instance = self.algorithm_class([IdentityTransformer()])
        elif self.algorithm_class is ParallelTransformer:
            instance = self.algorithm_class([("a", IdentityTransformer())])
        else:
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


class TestChainedTransformer:
    def test_simple_chaining(self):
        data = pd.DataFrame(np.ones((10, 3)), columns=list("abc")) * 2
        t = ChainedTransformer(transformer_chain=[FixedScaler(3, 1), FixedScaler(2)])
        make_action_safe(t.transform)(t, data)

        assert id(t.data) == id(data)
        assert_frame_equal(t.transformed_data_, (data - 1) / 3 / 2)

    def test_chaining_with_training(self):
        data = pd.DataFrame(np.ones((10, 3)), columns=list("abc"))
        train_data = pd.DataFrame(np.ones((10, 3)), columns=list("abc")) * 5
        # The first scaler is expected to learn the original data scale (5), the second scaler is ecpected to learn
        # the output of the first (3)
        t = ChainedTransformer([TrainableAbsMaxScaler(out_max=3), TrainableAbsMaxScaler()])
        make_optimize_safe(t.self_optimize)(t, [train_data])

        # Train results
        assert t.transformer_chain[0].data_max == 5
        assert t.transformer_chain[1].data_max == 3

        t.transform(data)

        assert_frame_equal(t.transformed_data_, data / 5)
