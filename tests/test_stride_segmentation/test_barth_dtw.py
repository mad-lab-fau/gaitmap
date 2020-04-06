import numpy as np
import pytest
from pandas.testing import assert_frame_equal
import pandas as pd

from gaitmap.base import BaseType
from gaitmap.stride_segmentation import BarthDtw, create_dtw_template
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin
from tests.test_stride_segmentation.test_base_dtw import (
    TestSimpleSegment,
    TestMultiDimensionalArrayInputs,
    TestMultiSensorInputs,
)


class TestMetaFunctionality(TestAlgorithmMixin):
    algorithm_class = BarthDtw
    __test__ = True

    @pytest.fixture()
    def after_action_instance(self) -> BaseType:
        template = create_dtw_template(np.array([0, 1.0, 0]), sampling_rate_hz=100.0)
        dtw = self.algorithm_class(template=template, max_cost=0.5, min_match_length=None, min_stride_time_s=None)
        data = np.array([0, 1.0, 0])
        dtw.segment(data, sampling_rate_hz=100)
        return dtw


class DtwTestBaseBarth:
    def init_dtw(self, template, **kwargs):
        defaults = dict(
            max_cost=0.5, min_match_length=None, min_stride_time_s=None, find_matches_method="min_under_thres"
        )
        kwargs = {**defaults, **kwargs}
        return BarthDtw(template=template, **kwargs)


class TestBarthDewAdditions(DtwTestBaseBarth):
    def test_stride_list(self):
        """Test that the output of the stride list is correct."""
        sequence = 2 * [*np.ones(5) * 2, 0, 1.0, 0, *np.ones(5) * 2]
        template = create_dtw_template(np.array([0, 1.0, 0]), sampling_rate_hz=100.0)
        dtw = self.init_dtw(template).segment(np.array(sequence), sampling_rate_hz=100.0,)

        expected_stride_list = pd.DataFrame(columns=["start", "end"])
        expected_stride_list["start"] = [5, 18]
        expected_stride_list["end"] = [7, 20]
        assert_frame_equal(dtw.stride_list_, expected_stride_list)

    def test_stride_list_multi_d(self):
        """Test that the output of the stride list is correct."""
        sensor1 = np.array([*np.ones(5) * 2, 0, 1.0, 0, *np.ones(5) * 2])
        sensor1 = pd.DataFrame(sensor1, columns=["col1"])
        sensor2 = np.array([*np.ones(2) * 2, 0, 1.0, 0, *np.ones(8) * 2])
        sensor2 = pd.DataFrame(sensor2, columns=["col1"])
        data = {"sensor1": sensor1, "sensor2": sensor2}

        template = [0, 1.0, 0]
        template = pd.DataFrame(template, columns=["col1"])
        template = create_dtw_template(template, sampling_rate_hz=100.0)
        dtw = self.init_dtw(template=template)

        dtw = dtw.segment(data=data, sampling_rate_hz=100)
        assert_frame_equal(dtw.stride_list_["sensor1"], pd.DataFrame([[5, 7]], columns=["start", "end"]))
        assert_frame_equal(dtw.stride_list_["sensor2"], pd.DataFrame([[2, 4]], columns=["start", "end"]))

    @pytest.mark.parametrize("min_stride_time_s", (None, 0.5))
    @pytest.mark.parametrize("min_match_length", (None, 4))
    @pytest.mark.parametrize("sampling_rate_hz", (100, 50))
    def test_min_stride_time(self, min_stride_time_s, min_match_length, sampling_rate_hz):
        """Test that min_stride_time is correctly converted into `min_match_length`.

        min_stride_time will always overwrite min_match_length
        """
        sequence = [*np.ones(5) * 2, 0, 1.0, 0, *np.ones(5) * 2]
        template = create_dtw_template(np.array([0, 1.0, 0]), sampling_rate_hz=100.0)
        dtw = self.init_dtw(template, min_stride_time_s=min_stride_time_s, min_match_length=min_match_length)

        # It will only converted during the segment:
        assert dtw.min_stride_time_s == min_stride_time_s
        assert dtw.min_match_length == min_match_length

        dtw.segment(np.array(sequence), sampling_rate_hz=sampling_rate_hz)
        if min_stride_time_s:
            assert dtw.min_stride_time_s == min_stride_time_s
            assert dtw.min_match_length == min_stride_time_s * sampling_rate_hz
        else:
            assert dtw.min_stride_time_s == min_stride_time_s
            assert dtw.min_match_length == min_match_length


# Add all the tests of base dtw, as they should pass here as well


class TestSimpleSegmentBarth(DtwTestBaseBarth, TestSimpleSegment):
    pass


class TestMultiDimensionalArrayInputsBarth(DtwTestBaseBarth, TestMultiDimensionalArrayInputs):
    pass


class TestMultiSensorInputsBarth(DtwTestBaseBarth, TestMultiSensorInputs):
    pass
