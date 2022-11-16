from typing import Type, Any, Dict, Callable

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pandas._testing import assert_frame_equal, assert_series_equal

from gaitmap.data_transform import (
    Resample,
    SlidingWindowMean,
    SlidingWindowStd,
    SlidingWindowVar,
    SlidingWindowGradient,
    BaseTransformer,
)
from gaitmap.data_transform._feature_transform import _PandasRollingFeatureTransform
from gaitmap.utils.consts import BF_COLS
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin

all_rolling_transformer = [
    (SlidingWindowMean, {"window_size_s": 1}, np.mean),
    (SlidingWindowStd, {"window_size_s": 1}, lambda _df, axis: _df.std(axis=axis)),
    (SlidingWindowVar, {"window_size_s": 1}, lambda _df, axis: _df.var(axis=axis)),
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

    @pytest.fixture(
        params=[*all_rolling_transformer, (SlidingWindowGradient, {"window_size_s": 1}, None)], autouse=True
    )
    def set_algo_class(self, request):
        self.algorithm_class, self.algo_params, _ = request.param

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data, healthy_example_stride_borders) -> BaseTransformer:
        data_left = healthy_example_imu_data["left_sensor"].iloc[:1000]
        data_left.columns = BF_COLS
        instance = self.algorithm_class(**self.algo_params)
        after_instance = instance.transform(data_left, sampling_rate_hz=100)
        return after_instance


@pytest.fixture(params=["df", "series"])
def df_or_series_imu_data(request, healthy_example_imu_data):
    """A ficture that returns either a df or a series of imu data."""
    data_left = healthy_example_imu_data["left_sensor"].iloc[:1000]
    data_left.columns = BF_COLS
    if request.param == "df":
        return data_left
    return data_left.iloc[:, 0]


class TestResample:
    def test_resample(self, df_or_series_imu_data):
        instance = Resample(target_sampling_rate_hz=10)
        after_instance = instance.transform(df_or_series_imu_data, sampling_rate_hz=100)
        assert after_instance.transformed_data_.shape[0] == 100
        assert after_instance.data is df_or_series_imu_data
        assert after_instance.sampling_rate_hz == 100
        assert hasattr(after_instance, "roi_list") is False
        assert hasattr(after_instance, "transformed_roi_list_") is False

    def test_resample_roi_list(self, healthy_example_stride_borders):
        in_stride_borders = healthy_example_stride_borders["left_sensor"]
        instance = Resample(target_sampling_rate_hz=10)
        after_instance = instance.transform(roi_list=in_stride_borders, sampling_rate_hz=100)
        assert after_instance.transformed_roi_list_.shape == in_stride_borders.shape
        assert_frame_equal(
            after_instance.transformed_roi_list_[["start", "end"]],
            (in_stride_borders[["start", "end"]] / 10.0).round().astype(int),
        )
        # Test that all other columns are the same
        assert_frame_equal(
            after_instance.transformed_roi_list_.drop(["start", "end"], axis=1),
            in_stride_borders.drop(["start", "end"], axis=1),
        )
        assert after_instance.roi_list is in_stride_borders
        assert after_instance.sampling_rate_hz == 100
        assert hasattr(after_instance, "transformed_data_") is False
        assert hasattr(after_instance, "data") is False

    def test_resample_data_and_roi_list(self, df_or_series_imu_data, healthy_example_stride_borders):
        in_stride_borders = healthy_example_stride_borders["left_sensor"]
        instance = Resample(target_sampling_rate_hz=10)
        after_instance = instance.transform(df_or_series_imu_data, roi_list=in_stride_borders, sampling_rate_hz=100)
        assert after_instance.transformed_data_.shape[0] == 100
        assert after_instance.transformed_roi_list_.shape == in_stride_borders.shape
        assert after_instance.roi_list is in_stride_borders
        assert after_instance.data is df_or_series_imu_data

    def test_require_sampling_rate(self):
        instance = Resample(target_sampling_rate_hz=10)
        with pytest.raises(ValueError) as e:
            instance.transform()

        assert "sampling_rate_hz" in str(e.value)

    def test_require_target_sampling_rate(self):
        instance = Resample()
        with pytest.raises(ValueError) as e:
            instance.transform(sampling_rate_hz=10)

        assert "target_sampling_rate_hz" in str(e.value)


class TestSlidingWindowTransformers:
    algorithm_class: Type[_PandasRollingFeatureTransform]
    algo_params: Dict[str, Any]
    equivalent_method: Callable

    @pytest.fixture(params=all_rolling_transformer, autouse=True)
    def set_algo_class(self, request):
        self.algorithm_class, self.algo_params, self.equivalent_method = request.param

    @pytest.mark.parametrize(
        "window_size_s,effective_win_size", [(1, 101), (0.5, 51), (0.1, 11), (0.23, 23), (0.111, 11)]
    )
    def test_effective_window_size_samples(self, healthy_example_imu_data, window_size_s, effective_win_size):
        data_left = healthy_example_imu_data["left_sensor"].iloc[:100]
        data_left.columns = BF_COLS
        instance = self.algorithm_class(window_size_s=window_size_s)
        after_instance = instance.transform(data_left, sampling_rate_hz=100)
        assert after_instance.effective_window_size_samples_ == effective_win_size

    def test_output(self, df_or_series_imu_data):
        """Test the output shape and the values for the first couple of windows."""
        win_size_s = 0.1
        instance = self.algorithm_class(window_size_s=win_size_s)
        after_instance = instance.transform(df_or_series_imu_data, sampling_rate_hz=100)

        # Test the shape
        assert after_instance.transformed_data_.shape == df_or_series_imu_data.shape
        # Test output column names
        if isinstance(df_or_series_imu_data, pd.DataFrame):
            assert after_instance.transformed_data_.columns.tolist() == [
                f"{self.algorithm_class._prefix}__{col}" for col in BF_COLS
            ]
        else:
            assert after_instance.transformed_data_.name == f"{self.algorithm_class._prefix}__{BF_COLS[0]}"
        # Assert value of first window
        assert_array_almost_equal(
            after_instance.transformed_data_.iloc[:1].to_numpy().flatten(),
            self.equivalent_method(pd.DataFrame(df_or_series_imu_data).iloc[:6], axis=0).to_numpy().flatten(),
        )

        # Assert the first full window
        assert_array_almost_equal(
            after_instance.transformed_data_.iloc[5:6].to_numpy().flatten(),
            self.equivalent_method(pd.DataFrame(df_or_series_imu_data).iloc[:11], axis=0).to_numpy().flatten(),
        )

        # Assert the last window
        assert_array_almost_equal(
            after_instance.transformed_data_.iloc[-1:].to_numpy().flatten(),
            self.equivalent_method(pd.DataFrame(df_or_series_imu_data).iloc[-6:], axis=0).to_numpy().flatten(),
        )

        # Test that data and sampling rate are set as attributes
        assert after_instance.data is df_or_series_imu_data
        assert after_instance.sampling_rate_hz == 100
