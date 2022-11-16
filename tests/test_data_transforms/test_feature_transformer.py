import pytest
from pandas._testing import assert_frame_equal

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


class TestResample:
    @pytest.fixture(params=["df", "series"])
    def df_or_series_imu_data(self, request, healthy_example_imu_data):
        """A ficture that returns either a df or a series of imu data."""
        data_left = healthy_example_imu_data["left_sensor"].iloc[:1000]
        data_left.columns = BF_COLS
        if request.param == "df":
            return data_left
        return data_left.iloc[:, 0]

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
