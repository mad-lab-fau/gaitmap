from gaitmap.event_detection.rampp_event_detection import RamppEventDetection, _detect_min_vel
from gaitmap.utils.consts import *

import pytest
import pandas as pd
from numpy.testing import assert_array_equal
import numpy as np


class TestEventDetectionRampp:
    """Test the event detection by Rampp."""

    def test_valid_input_data(self, healthy_example_stride_borders):
        """Test if error is raised correctly on invalid input data type"""
        data = pd.DataFrame({"a": [0, 1, 2], "b": [3, 4, 5]})
        ed = RamppEventDetection()
        with pytest.raises(ValueError, match=r"Provided data set is not supported by gaitmap"):
            ed.detect(data, 204.8, healthy_example_stride_borders)

    def test_min_vel_search_wind_size_dummy_data(self):
        """Test if error is raised correctly if windows size matches the size of the input data"""
        dummy_gyr = np.ones((100, 3))
        with pytest.raises(ValueError, match=r"The value chosen for min_vel_search_wind_size is too large*"):
            _detect_min_vel(dummy_gyr, dummy_gyr.size)

    def test_valid_min_vel_search_wind_size(self, healthy_example_imu_data, healthy_example_stride_borders):
        """Test if error is raised correctly on too large min_vel_search_wind_size"""
        data_left = healthy_example_imu_data["left_sensor"]
        data_left.columns = BF_COLS
        stride_list_left = healthy_example_stride_borders["left_sensor"]
        ed = RamppEventDetection(min_vel_search_wind_size=5000)
        with pytest.raises(ValueError, match=r"The value chosen for min_vel_search_wind_size is too large*"):
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
