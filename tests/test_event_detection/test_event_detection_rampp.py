import random
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas._testing import assert_frame_equal

from gaitmap.base import BaseType
from gaitmap.event_detection.rampp_event_detection import RamppEventDetection, _detect_min_vel, _detect_tc
from gaitmap.utils import coordinate_conversion, datatype_helper
from gaitmap.utils.consts import BF_COLS
from gaitmap.utils.exceptions import ValidationError
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin
from tests.mixins.test_caching_mixin import TestCachingMixin


class MetaTestConfig:
    algorithm_class = RamppEventDetection

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data, healthy_example_stride_borders) -> BaseType:
        data_left = healthy_example_imu_data["left_sensor"]
        data_left.columns = BF_COLS
        # only use the first entry of the stride list
        stride_list_left = healthy_example_stride_borders["left_sensor"].iloc[0:1]
        ed = RamppEventDetection()
        ed.detect(data_left, stride_list_left, 204.8)
        return ed


class TestMetaFunctionality(MetaTestConfig, TestAlgorithmMixin):
    __test__ = True


class TestCachingFunctionality(MetaTestConfig, TestCachingMixin):
    __test__ = True


class TestEventDetectionRampp:
    """Test the event detection by Rampp."""

    def test_multi_sensor_input(self, healthy_example_imu_data, healthy_example_stride_borders, snapshot):
        """Dummy test to see if the algorithm is generally working on the example data"""
        data = coordinate_conversion.convert_to_fbf(
            healthy_example_imu_data, left=["left_sensor"], right=["right_sensor"]
        )

        ed = RamppEventDetection()
        ed.detect(data, healthy_example_stride_borders, 204.8)

        snapshot.assert_match(ed.min_vel_event_list_["left_sensor"], "left", check_dtype=False)
        snapshot.assert_match(ed.min_vel_event_list_["right_sensor"], "right", check_dtype=False)
        snapshot.assert_match(ed.segmented_event_list_["left_sensor"], "left_segmented", check_dtype=False)
        snapshot.assert_match(ed.segmented_event_list_["right_sensor"], "right_segmented", check_dtype=False)

    @pytest.mark.parametrize("var1, output", ((True, 1), (False, 0)))
    def test_postprocessing(self, healthy_example_imu_data, healthy_example_stride_borders, var1, output):
        data_left = healthy_example_imu_data["left_sensor"]
        data_left.columns = BF_COLS
        # only use the first entry of the stride list
        stride_list_left = healthy_example_stride_borders["left_sensor"].iloc[0:1]

        def mock_func(event_list, *args, **kwargs):
            return event_list, None

        ed = RamppEventDetection(enforce_consistency=var1)
        with patch("gaitmap.event_detection._base.enforce_stride_list_consistency", side_effect=mock_func) as mock:
            ed.detect(data_left, stride_list_left, 204.8)

        assert mock.call_count == output

    @pytest.mark.parametrize("enforce_consistency, output", ((False, False), (True, True)))
    def test_disable_min_vel_event_list(
        self, healthy_example_imu_data, healthy_example_stride_borders, enforce_consistency, output
    ):
        data_left = healthy_example_imu_data["left_sensor"]
        data_left.columns = BF_COLS
        # only use the first entry of the stride list
        stride_list_left = healthy_example_stride_borders["left_sensor"].iloc[0:1]

        ed = RamppEventDetection(enforce_consistency=enforce_consistency)
        ed.detect(data_left, stride_list_left, 204.8)

        assert hasattr(ed, "min_vel_event_list_") == output

    def test_multi_sensor_input_dict(self, healthy_example_imu_data, healthy_example_stride_borders):
        """Test to see if the algorithm is generally working on the example data when provided as dict"""
        data = coordinate_conversion.convert_to_fbf(
            healthy_example_imu_data, left=["left_sensor"], right=["right_sensor"]
        )

        dict_keys = ["l", "r"]
        data_dict = {dict_keys[0]: data["left_sensor"], dict_keys[1]: data["right_sensor"]}
        stride_list_dict = {
            dict_keys[0]: healthy_example_stride_borders["left_sensor"],
            dict_keys[1]: healthy_example_stride_borders["right_sensor"],
        }

        ed = RamppEventDetection()
        ed.detect(data_dict, stride_list_dict, 204.8)

        assert list(datatype_helper.get_multi_sensor_names(ed.min_vel_event_list_)) == dict_keys
        assert list(datatype_helper.get_multi_sensor_names(ed.segmented_event_list_)) == dict_keys

    def test_equal_output_dict_df(self, healthy_example_imu_data, healthy_example_stride_borders):
        """Test if output is similar for input dicts or regular multisensor data sets"""
        data = coordinate_conversion.convert_to_fbf(
            healthy_example_imu_data, left=["left_sensor"], right=["right_sensor"]
        )

        ed_df = RamppEventDetection()
        ed_df.detect(data, healthy_example_stride_borders, 204.8)

        dict_keys = ["l", "r"]
        data_dict = {dict_keys[0]: data["left_sensor"], dict_keys[1]: data["right_sensor"]}
        stride_list_dict = {
            dict_keys[0]: healthy_example_stride_borders["left_sensor"],
            dict_keys[1]: healthy_example_stride_borders["right_sensor"],
        }

        ed_dict = RamppEventDetection()
        ed_dict.detect(data_dict, stride_list_dict, 204.8)

        assert_frame_equal(ed_df.min_vel_event_list_["left_sensor"], ed_dict.min_vel_event_list_["l"])
        assert_frame_equal(ed_df.min_vel_event_list_["right_sensor"], ed_dict.min_vel_event_list_["r"])

    def test_valid_input_data(self, healthy_example_stride_borders):
        """Test if error is raised correctly on invalid input data type"""
        data = pd.DataFrame({"a": [0, 1, 2], "b": [3, 4, 5]})
        ed = RamppEventDetection()
        with pytest.raises(ValidationError) as e:
            ed.detect(data, healthy_example_stride_borders, 204.8)

        assert "The passed object appears to be neither single- or multi-sensor data" in str(e)

    def test_min_vel_search_win_size_ms_dummy_data(self):
        """Test if error is raised correctly if windows size matches the size of the input data"""
        dummy_gyr = np.ones((100, 3))
        with pytest.raises(ValueError, match=r"min_vel_search_win_size_ms is*"):
            _detect_min_vel(dummy_gyr, dummy_gyr.size)

    def test_valid_min_vel_search_win_size_ms(self, healthy_example_imu_data, healthy_example_stride_borders):
        """Test if error is raised correctly on too large min_vel_search_win_size_ms"""
        data_left = healthy_example_imu_data["left_sensor"]
        data_left = coordinate_conversion.convert_left_foot_to_fbf(data_left)
        stride_list_left = healthy_example_stride_borders["left_sensor"]
        ed = RamppEventDetection(min_vel_search_win_size_ms=5000)
        with pytest.raises(ValueError, match=r"min_vel_search_win_size_ms is*"):
            ed.detect(data_left, stride_list_left, 204.8)

    def test_valid_ic_search_region_ms(self, healthy_example_imu_data, healthy_example_stride_borders):
        """Test if error is raised correctly on too small ic_search_region_ms"""
        data_left = healthy_example_imu_data["left_sensor"]
        data_left = coordinate_conversion.convert_left_foot_to_fbf(data_left)
        stride_list_left = healthy_example_stride_borders["left_sensor"]
        ed = RamppEventDetection(ic_search_region_ms=(1, 1))
        with pytest.raises(ValueError):
            ed.detect(data_left, stride_list_left, 204.8)

    def test_input_stride_list_size_one(self, healthy_example_imu_data, healthy_example_stride_borders):
        """Test if gait event detection also works with stride list of length 1"""
        data_left = healthy_example_imu_data["left_sensor"]
        data_left = coordinate_conversion.convert_left_foot_to_fbf(data_left)
        # only use the first entry of the stride list
        stride_list_left = healthy_example_stride_borders["left_sensor"].iloc[0:1]
        ed = RamppEventDetection()
        ed.detect(data_left, stride_list_left, 204.8)
        # per default min_vel_event_list_ has 6 columns
        assert_array_equal(np.array(ed.min_vel_event_list_.shape[1]), 6)
        # per default segmented_event_list_ has 5 columns
        assert_array_equal(np.array(ed.segmented_event_list_.shape[1]), 5)

    def test_correct_s_id(self, healthy_example_imu_data, healthy_example_stride_borders):
        """Test if the s_id from the stride list is correctly transferred to the output of event detection"""
        data_left = healthy_example_imu_data["left_sensor"]
        data_left = coordinate_conversion.convert_left_foot_to_fbf(data_left)
        stride_list_left = healthy_example_stride_borders["left_sensor"]
        # switch s_ids in stride list to random numbers
        stride_list_left["s_id"] = random.sample(range(1000), stride_list_left["s_id"].size)
        ed = RamppEventDetection()
        ed.detect(data_left, stride_list_left, 204.8)

        # Check that all of the old stride ids are still in the new one
        assert np.all(ed.min_vel_event_list_.index.isin(stride_list_left["s_id"]))
        assert np.all(ed.segmented_event_list_.index.isin(stride_list_left["s_id"]))
        # The new start should be inside the old stride
        combined = pd.merge(ed.min_vel_event_list_, stride_list_left, on="s_id")
        assert np.all(combined["min_vel"] < combined["end_y"])
        assert np.all(combined["min_vel"] > combined["start_y"])
        # The new starts and ends should be identical to the old ones
        combined = pd.merge(ed.segmented_event_list_, stride_list_left, on="s_id")
        assert np.all(combined["start_x"] == combined["start_y"])
        assert np.all(combined["end_x"] == combined["end_y"])

    def test_single_data_multi_stride_list(self, healthy_example_imu_data, healthy_example_stride_borders):
        """Test correct error for combination of single sensor data set and multi sensor stride list"""
        data_left = healthy_example_imu_data["left_sensor"]
        data_left = coordinate_conversion.convert_left_foot_to_fbf(data_left)
        stride_list_left = healthy_example_stride_borders
        ed = RamppEventDetection()
        with pytest.raises(ValidationError):
            ed.detect(data_left, 204.8, stride_list_left)

    def test_multi_data_single_stride_list(self, healthy_example_imu_data, healthy_example_stride_borders):
        """Test correct error for combination of multi sensor data set and single sensor stride list"""
        data_left = healthy_example_imu_data["left_sensor"]
        data_left = coordinate_conversion.convert_left_foot_to_fbf(data_left)
        stride_list_left = healthy_example_stride_borders
        ed = RamppEventDetection()
        with pytest.raises(ValidationError):
            ed.detect(data_left, 204.8, stride_list_left)

    def test_sign_change_for_detect_tc(self):
        """Test correct handling of signal that does or does not provide a change of the sign"""

        # with sign change
        signal1 = np.concatenate([np.ones(10), np.ones(10) * -1])
        assert _detect_tc(signal1) == 9

        # without sign change
        signal2 = np.ones(10)
        assert np.isnan(_detect_tc(signal2))
