import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from gaitmap.base import BaseType
from gaitmap.stride_segmentation import BarthDtw, create_dtw_template
from gaitmap.utils.coordinate_conversion import convert_to_fbf
from gaitmap.utils.dataset_helper import is_single_sensor_stride_list, is_multi_sensor_stride_list
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
    def test_real_data_both_feed_regression(self, healthy_example_imu_data, snapshot):
        data = convert_to_fbf(healthy_example_imu_data, right=["right_sensor"], left=["left_sensor"])
        dtw = BarthDtw()  # Test with default paras
        dtw.segment(data, sampling_rate_hz=204.8)

        snapshot.assert_match(dtw.stride_list_["left_sensor"], "left")
        snapshot.assert_match(dtw.stride_list_["right_sensor"], "right")

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

    def test_conflict_resolution_on_off(self, healthy_example_imu_data):
        data = convert_to_fbf(healthy_example_imu_data, right=["right_sensor"], left=["left_sensor"])
        # For both cases set the threshold so high that wrong matches will occure
        max_cost = 2500
        min_match_length_s = 0.1
        # off
        dtw = BarthDtw(max_cost=max_cost, conflict_resolution=False, min_match_length_s=min_match_length_s)
        dtw.segment(data, sampling_rate_hz=204.8)
        no_conflict_resolution = dtw.matches_start_end_["left_sensor"]

        # Validate that there are no overlaps
        assert np.any(np.diff(no_conflict_resolution.flatten()) < 0)

        # on
        dtw = BarthDtw(max_cost=max_cost, conflict_resolution=True, min_match_length_s=min_match_length_s)
        dtw.segment(data, sampling_rate_hz=204.8)
        conflict_resolution = dtw.matches_start_end_["left_sensor"]

        # Validate that there are indeed overlaps
        assert not np.any(np.diff(conflict_resolution.flatten()) < 0)
        assert len(conflict_resolution) == 29


class DtwTestBaseBarth:
    def init_dtw(self, template, **kwargs):
        defaults = dict(
            max_cost=0.5, min_match_length_s=None, find_matches_method="min_under_thres", snap_to_min_win_ms=None
        )
        kwargs = {**defaults, **kwargs}
        return BarthDtw(template=template, **kwargs)


class TestBarthDtwAdditions(DtwTestBaseBarth):
    def test_stride_list(self):
        """Test that the output of the stride list is correct."""
        sequence = 2 * [*np.ones(5) * 2, 0, 1.0, 0, *np.ones(5) * 2]
        template = create_dtw_template(np.array([0, 1.0, 0]), sampling_rate_hz=100.0)
        dtw = self.init_dtw(template).segment(np.array(sequence), sampling_rate_hz=100.0)

        expected_stride_list = pd.DataFrame(columns=["s_id", "start", "end"])
        expected_stride_list["start"] = [5, 18]
        expected_stride_list["end"] = [7, 20]
        expected_stride_list["s_id"] = [0, 1]
        assert_frame_equal(dtw.stride_list_.astype(np.int64), expected_stride_list.astype(np.int64))

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
        assert_frame_equal(
            dtw.stride_list_["sensor1"].astype(np.int64),
            pd.DataFrame([[0, 5, 7]], columns=["s_id", "start", "end"]).astype(np.int64),
        )
        assert_frame_equal(
            dtw.stride_list_["sensor2"].astype(np.int64),
            pd.DataFrame([[0, 2, 4]], columns=["s_id", "start", "end"]).astype(np.int64),
        )

    def test_stride_list_passes_test_func(self):
        sequence = 2 * [*np.ones(5) * 2, 0, 1.0, 0, *np.ones(5) * 2]
        template = create_dtw_template(np.array([0, 1.0, 0]), sampling_rate_hz=100.0)
        dtw = self.init_dtw(template).segment(np.array(sequence), sampling_rate_hz=100.0)

        assert is_single_sensor_stride_list(dtw.stride_list_)

    def test_stride_list_passes_test_func_multiple(self):
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

        assert is_multi_sensor_stride_list(dtw.stride_list_)


class TestPostProcessing:
    """The postprocessing can get quite complex. This tries to split it in a couple of unit tests.

    Snapping is not tested here as it is well covered by the regression tests
    """

    def test_simple_stride_time(self):
        example_stride_list = np.array([np.arange(10), np.arange(10) + 1.0]).T
        bad_strides = [2, 5, 9]
        example_stride_list[bad_strides, 1] -= 0.5
        to_keep = np.ones(len(example_stride_list)).astype(bool)

        # Disable all other postprocessing
        dtw = BarthDtw(snap_to_min_win_ms=None, conflict_resolution=False)
        # Set min threshold to 0.6
        dtw._min_sequence_length = 0.6

        start_end, to_keep = dtw._postprocess_matches(
            None, [], np.array([]), matches_start_end=example_stride_list, to_keep=to_keep
        )

        # Check that start-end is unmodified
        assert_array_equal(example_stride_list, start_end)
        # Check that 3 strides were identified for removal
        assert np.sum(~to_keep) == 3
        # Check that the correct 3 strides were identified
        assert np.all(~to_keep[bad_strides])

    def test_simple_double_start(self):
        example_stride_list = np.array([np.arange(10), np.arange(10) + 1.0]).T
        cost = np.ones(len(example_stride_list))
        # Introduce errors
        example_stride_list[4, 0] = example_stride_list[3, 0]
        cost[3] = 10
        cost[4] = 5  # 4 should be selected
        example_stride_list[8, 0] = example_stride_list[7, 0]
        example_stride_list[9, 0] = example_stride_list[7, 0]
        cost[7] = 5  # 7 should be selected
        cost[8] = 10
        cost[9] = 10
        bad_strides = [3, 8, 9]
        to_keep = np.ones(len(example_stride_list)).astype(bool)

        # Disable all other postprocessing
        dtw = BarthDtw(snap_to_min_win_ms=None, conflict_resolution=True, min_match_length_s=None)
        # Set threshold to None
        dtw._min_sequence_length = None

        start_end, to_keep = dtw._postprocess_matches(
            None, [], cost=cost, matches_start_end=example_stride_list, to_keep=to_keep
        )

        # Check that start-end is unmodified
        assert_array_equal(example_stride_list, start_end)
        # Check that 3 strides were identified for removal
        assert np.sum(~to_keep) == 3
        # Check that the correct 3 strides were identified
        assert np.all(~to_keep[bad_strides])

    def test_previous_removal_double_start(self):
        example_stride_list = np.array([np.arange(10), np.arange(10) + 1.0]).T
        cost = np.ones(len(example_stride_list))
        # Introduce errors
        example_stride_list[4, 0] = example_stride_list[3, 0]
        cost[3] = 10
        cost[4] = 5  # 4 should be selected
        example_stride_list[8, 0] = example_stride_list[7, 0]
        example_stride_list[9, 0] = example_stride_list[7, 0]
        cost[7] = 5  # 7 should be selected
        cost[8] = 10
        cost[9] = 2  # Has the lowest cost, but should be removed as it is a short stride
        bad_strides = [3, 8, 9]
        # introduce additional short strides
        # 9 will already be removed because of short stride
        short_strides = [1, 5, 9]
        example_stride_list[short_strides, 1] = example_stride_list[short_strides, 0] + 0.5

        bad_strides = set(bad_strides)
        bad_strides.update(short_strides)
        bad_strides = list(bad_strides)
        to_keep = np.ones(len(example_stride_list)).astype(bool)

        # Disable all other postprocessing
        dtw = BarthDtw(snap_to_min_win_ms=None, conflict_resolution=True)
        # Set min threshold to 0.6
        dtw._min_sequence_length = 0.6

        start_end, to_keep = dtw._postprocess_matches(
            None, [], cost=cost, matches_start_end=example_stride_list, to_keep=to_keep
        )

        # Check that start-end is unmodified
        assert_array_equal(example_stride_list, start_end)
        # Check that 5 strides were identified for removal
        assert np.sum(~to_keep) == 5
        # Check that the correct 5 strides were identified
        assert np.all(~to_keep[bad_strides])

    def test_previous_removal_double_start_unsorted(self):
        example_stride_list = np.array([np.arange(10), np.arange(10) + 1.0]).T
        cost = np.ones(len(example_stride_list))
        # Introduce errors
        example_stride_list[4, 0] = example_stride_list[3, 0]
        cost[3] = 10
        cost[4] = 5  # 4 should be selected
        example_stride_list[8, 0] = example_stride_list[2, 0]
        example_stride_list[9, 0] = example_stride_list[2, 0]
        cost[2] = 5  # 2 should be selected event though the structure was unsorted
        cost[8] = 10
        cost[9] = 2  # Has the lowest cost, but should be removed as it is a short stride
        bad_strides = [3, 8, 9]
        # introduce additional short strides
        # 9 will already be removed because of short stride
        short_strides = [1, 5, 9]
        example_stride_list[short_strides, 1] = example_stride_list[short_strides, 0] + 0.5

        bad_strides = set(bad_strides)
        bad_strides.update(short_strides)
        bad_strides = list(bad_strides)
        to_keep = np.ones(len(example_stride_list)).astype(bool)

        # Disable all other postprocessing
        dtw = BarthDtw(snap_to_min_win_ms=None, conflict_resolution=True)
        # Set min threshold to 0.6
        dtw._min_sequence_length = 0.6

        start_end, to_keep = dtw._postprocess_matches(
            None, [], cost=cost, matches_start_end=example_stride_list, to_keep=to_keep
        )

        # Check that start-end is unmodified
        assert_array_equal(example_stride_list, start_end)
        # Check that 5 strides were identified for removal
        assert np.sum(~to_keep) == 5
        # Check that the correct 5 strides were identified
        assert np.all(~to_keep[bad_strides])


# Add all the tests of base dtw, as they should pass here as well


class TestSimpleSegmentBarth(DtwTestBaseBarth, TestSimpleSegment):
    pass


class TestMultiDimensionalArrayInputsBarth(DtwTestBaseBarth, TestMultiDimensionalArrayInputs):
    pass


class TestMultiSensorInputsBarth(DtwTestBaseBarth, TestMultiSensorInputs):
    pass
