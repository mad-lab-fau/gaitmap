import numpy as np
import pandas as pd
import pytest
from mixins.test_algorithm_mixin import TestAlgorithmMixin
from numpy.testing import assert_equal
from scipy.signal import butter, sosfiltfilt

from gaitmap.data_transform import BaseFilter, ButterworthFilter
from gaitmap.utils.consts import BF_COLS


class TestButterworthMetaFunctionality(TestAlgorithmMixin):
    __test__ = True

    algorithm_class = ButterworthFilter

    def test_empty_init(self):
        pytest.skip()

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data, healthy_example_stride_borders) -> BaseFilter:
        data_left = healthy_example_imu_data["left_sensor"].iloc[:100]
        data_left.columns = BF_COLS
        instance = self.algorithm_class(1, 30)
        after_instance = instance.transform(data_left, sampling_rate_hz=100)
        return after_instance


class TestButterworth:
    @pytest.mark.parametrize("in_val", (pd.DataFrame(np.random.rand(50, 3)), pd.Series(np.random.rand(50))))
    def test_input_type_and_shape_conserved(self, in_val):
        filter = ButterworthFilter(1, 5)

        before_dtype = type(in_val)
        before_shape = in_val.shape

        filter.transform(in_val, sampling_rate_hz=100)

        assert filter.transformed_data_.shape == before_shape
        assert isinstance(filter.transformed_data_, before_dtype)

        # We test for all shapes that they also work after `to_numpy`
        in_val = in_val.to_numpy()
        before_dtype = type(in_val)
        before_shape = in_val.shape

        filter.transform(in_val, sampling_rate_hz=100)

        assert filter.transformed_data_.shape == before_shape
        assert isinstance(filter.transformed_data_, before_dtype)

    def test_filter_is_applied_correctly(self):
        filter = ButterworthFilter(1, 5, "highpass")
        data = pd.DataFrame(np.random.rand(50, 3))
        sampling_rate_hz = 100

        # Filter the data in a traditional method
        filtered = sosfiltfilt(butter(1, 5, "highpass", output="sos", fs=sampling_rate_hz), x=data, axis=0)

        assert_equal(filter.filter(data, sampling_rate_hz=100).filtered_data_.to_numpy(), filtered)

        assert filter.transformed_data_ is filter.filtered_data_
