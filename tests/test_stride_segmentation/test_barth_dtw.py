import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from gaitmap.base import BaseType
from gaitmap.stride_segmentation import BarthDtw, create_dtw_template
from gaitmap.utils.coordinate_conversion import convert_to_fbf
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
        dtw = self.algorithm_class(template=template, max_cost=0.5, min_match_length_s=None)
        data = np.array([0, 1.0, 0])
        dtw.segment(data, sampling_rate_hz=100)
        return dtw


class TestRegressionOnRealData:
    # TODO: More real regression test needed
    def test_real_data_both_feed(self, healthy_example_imu_data):
        data = convert_to_fbf(healthy_example_imu_data, right=["right_sensor"], left=["left_sensor"])
        dtw = BarthDtw()  # Test with default paras
        dtw.segment(data, sampling_rate_hz=204.8)

        # For now only evaluate that the number of strides is correct
        assert len(dtw.stride_list_["left_sensor"]) == 28
        assert len(dtw.stride_list_["right_sensor"]) == 28

    def test_snapping_on_off(self, healthy_example_imu_data):
        data = convert_to_fbf(healthy_example_imu_data, right=["right_sensor"], left=["left_sensor"])
        # off
        dtw = BarthDtw(snap_to_min_win_ms=None)
        dtw.segment(data, sampling_rate_hz=204.8)
        out_without_snapping = dtw.matches_start_end_["left_sensor"]

        assert dtw.stride_list_["left_sensor"]["start"][0] == 364
        assert dtw.stride_list_["left_sensor"]["end"][0] == 574
        assert_array_equal(dtw.matches_start_end_["left_sensor"], dtw.matches_start_end_original_["left_sensor"])

        # on
        dtw = BarthDtw()  # Test with default paras
        dtw.segment(data, sampling_rate_hz=204.8)
        assert dtw.stride_list_["left_sensor"]["start"][0] == 364
        assert dtw.stride_list_["left_sensor"]["end"][0] == 584
        assert not np.array_equal(dtw.matches_start_end_["left_sensor"], dtw.matches_start_end_original_["left_sensor"])
        assert_array_equal(dtw.matches_start_end_original_["left_sensor"], out_without_snapping)


class DtwTestBaseBarth:
    def init_dtw(self, template, **kwargs):
        defaults = dict(
            max_cost=0.5, min_match_length_s=None, find_matches_method="min_under_thres", snap_to_min_win_ms=None
        )
        kwargs = {**defaults, **kwargs}
        return BarthDtw(template=template, **kwargs)


class TestBarthDewAdditions(DtwTestBaseBarth):
    def test_stride_list(self):
        """Test that the output of the stride list is correct."""
        sequence = 2 * [*np.ones(5) * 2, 0, 1.0, 0, *np.ones(5) * 2]
        template = create_dtw_template(np.array([0, 1.0, 0]), sampling_rate_hz=100.0)
        dtw = self.init_dtw(template).segment(np.array(sequence), sampling_rate_hz=100.0)

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


# Add all the tests of base dtw, as they should pass here as well


class TestSimpleSegmentBarth(DtwTestBaseBarth, TestSimpleSegment):
    pass


class TestMultiDimensionalArrayInputsBarth(DtwTestBaseBarth, TestMultiDimensionalArrayInputs):
    pass


class TestMultiSensorInputsBarth(DtwTestBaseBarth, TestMultiSensorInputs):
    pass
