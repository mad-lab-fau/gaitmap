from gaitmap.event_detection.rampp_event_detection import (
    RamppEventDetection,
    _detect_min_vel,
    _find_breaks_in_stride_list,
)
from gaitmap.utils.consts import *
from gaitmap.utils import coordinate_conversion

import pytest
import pandas as pd
from numpy.testing import assert_array_equal
import numpy as np

# TODO add meta tests


class TestEventDetectionRampp:
    """Test the event detection by Rampp."""

    # TODO add tests for multiple sensors and checks for input data / stride lists

    def test_single_sensor_input(self, healthy_example_imu_data, healthy_example_stride_borders):
        """Dummy test to see if the algorithm is generally working on the example data"""
        # TODO add assert statement / regression test to check against previous result
        data_left = healthy_example_imu_data["left_sensor"]
        data_left.columns = BF_COLS
        stride_list_left = healthy_example_stride_borders["left_sensor"]
        stride_list_left = stride_list_left.iloc[[0, 1, 2, 4, 5, 6]]

        ed = RamppEventDetection()
        ed.detect(data_left, 204.8, stride_list_left)

        return None

    def test_valid_input_data(self, healthy_example_stride_borders):
        """Test if error is raised correctly on invalid input data type"""
        data = pd.DataFrame({"a": [0, 1, 2], "b": [3, 4, 5]})
        ed = RamppEventDetection()
        with pytest.raises(ValueError, match=r"Provided data set is not supported by gaitmap"):
            ed.detect(data, 204.8, healthy_example_stride_borders)

    def test_min_vel_search_win_size_ms_dummy_data(self):
        """Test if error is raised correctly if windows size matches the size of the input data"""
        dummy_gyr = np.ones((100, 3))
        with pytest.raises(ValueError, match=r"The value chosen for min_vel_search_win_size_ms is too large*"):
            _detect_min_vel(dummy_gyr, dummy_gyr.size)

    def test_valid_min_vel_search_win_size_ms(self, healthy_example_imu_data, healthy_example_stride_borders):
        """Test if error is raised correctly on too large min_vel_search_win_size_ms"""
        data_left = healthy_example_imu_data["left_sensor"]
        data_left.columns = BF_COLS
        stride_list_left = healthy_example_stride_borders["left_sensor"]
        ed = RamppEventDetection(min_vel_search_win_size_ms=5000)
        with pytest.raises(ValueError, match=r"The value chosen for min_vel_search_win_size_ms is too large*"):
            ed.detect(data_left, 204.8, stride_list_left)

    def test_input_stride_list_size_one(self, healthy_example_imu_data, healthy_example_stride_borders):
        """Test if gait event detection also works with stride list of length 1"""
        data_left = healthy_example_imu_data["left_sensor"]
        data_left.columns = BF_COLS
        # only use the first entry of the stride list
        stride_list_left = healthy_example_stride_borders["left_sensor"].iloc[0:1]
        ed = RamppEventDetection()
        ed.detect(data_left, 204.8, stride_list_left)
        # per default stride_events_ has 7 columns
        assert_array_equal(np.array(ed.stride_events_.shape), np.array((0, 7)))

    def test_correct_s_id(self, healthy_example_imu_data, healthy_example_stride_borders):
        """Test if the s_id from the stride list is correctly transferred to the output of event detection"""
        data_left = healthy_example_imu_data["left_sensor"]
        data_left = coordinate_conversion.convert_left_foot_to_fbf(data_left)
        stride_list_left = healthy_example_stride_borders["left_sensor"]
        # switch s_ids in stride list to random numbers
        stride_list_left["s_id"] = np.random.randint(1000, size=(stride_list_left["s_id"].size, 1))
        ed = RamppEventDetection()
        ed.detect(data_left, 204.8, stride_list_left)
        # find breaks in segmented strides and drop first strides of new sequences
        stride_list_breaks = _find_breaks_in_stride_list(stride_list_left)
        stride_list_left = stride_list_left.drop(stride_list_left.index[stride_list_breaks])
        # the s_ids of the event detection and the stride list should be identical (except for the last entry)
        assert_array_equal(np.array(ed.stride_events_["s_id"]), np.array(stride_list_left["s_id"])[:-1])
