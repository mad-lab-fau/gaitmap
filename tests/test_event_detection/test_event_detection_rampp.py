import random
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas._testing import assert_frame_equal

from gaitmap._event_detection_common._event_detection_mixin import _detect_min_vel_gyr_energy
from gaitmap.base import BaseType
from gaitmap.event_detection import RamppEventDetection
from gaitmap.utils import coordinate_conversion, datatype_helper
from gaitmap.utils.consts import BF_COLS
from gaitmap.utils.exceptions import ValidationError
from gaitmap_mad.event_detection._rampp_event_detection import _detect_tc_for_segmented_stride, _detect_tc_for_ic_stride
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin
from tests.mixins.test_caching_mixin import TestCachingMixin


common_arguments =pytest.mark.parametrize(
        ("input_stride_type", "imu_data", "stride_borders", "sampling_rate"),
        [
            ("segmented", "healthy_example_imu_data", "healthy_example_stride_borders", 204.8),
            ("ic", "healthy_example_imu_data_ic_stride", "healthy_example_stride_borders_ic_stride", 102.4),
        ],
    )
class MetaTestConfig:
    algorithm_class = RamppEventDetection

    @common_arguments
    def after_action_instance(self, input_stride_type, imu_data, stride_borders, sampling_rate, request) -> BaseType:
        data_left = request.getfixturevalue(imu_data)["left_sensor"]
        data_left.columns = BF_COLS
        # only use the first entry of the stride list
        stride_list_left = request.getfixturevalue(stride_borders)["left_sensor"].iloc[0:1]
        ed = RamppEventDetection(input_stride_type=input_stride_type)
        ed.detect(data_left, stride_list_left, sampling_rate_hz=sampling_rate)
        return ed


class TestMetaFunctionality(MetaTestConfig, TestAlgorithmMixin):
    __test__ = True


class TestCachingFunctionality(MetaTestConfig, TestCachingMixin):
    __test__ = True


class TestEventDetectionRampp:
    """Test the event detection by Rampp."""

    algorithm_class = RamppEventDetection

    @common_arguments
    def test_multi_sensor_input(self, input_stride_type, imu_data, stride_borders, sampling_rate, snapshot, request):
        """Dummy test to see if the algorithm is generally working on the example data."""
        data = coordinate_conversion.convert_to_fbf(
            request.getfixturevalue(imu_data), left=["left_sensor"], right=["right_sensor"]
        )

        ed = self.algorithm_class(input_stride_type=input_stride_type)
        ed.detect(data, request.getfixturevalue(stride_borders), sampling_rate_hz=sampling_rate)

        snapshot.assert_match(ed.min_vel_event_list_["left_sensor"], "left", check_dtype=False)
        snapshot.assert_match(ed.min_vel_event_list_["right_sensor"], "right", check_dtype=False)
        snapshot.assert_match(ed.segmented_event_list_["left_sensor"], "left_segmented", check_dtype=False)
        snapshot.assert_match(ed.segmented_event_list_["right_sensor"], "right_segmented", check_dtype=False)

    @pytest.mark.parametrize(("var1", "output"), ((True, 2), (False, 0)))
    def test_postprocessing_segmented_stride(
        self, healthy_example_imu_data, healthy_example_stride_borders, var1, output
    ):
        """Test postprocessing for "segmented" input_stride_type."""
        data_left = healthy_example_imu_data["left_sensor"]
        data_left.columns = BF_COLS
        # only use the first entry of the stride list
        stride_list_left = healthy_example_stride_borders["left_sensor"].iloc[0:1]

        def mock_func(event_list, *args, **kwargs):
            return event_list, None

        ed = self.algorithm_class(enforce_consistency=var1)
        with patch(
            "gaitmap._event_detection_common._event_detection_mixin.enforce_stride_list_consistency",
            side_effect=mock_func,
        ) as mock:
            ed.detect(data_left, stride_list_left, sampling_rate_hz=204.8)

        assert mock.call_count == output

        @pytest.mark.parametrize(("var1", "output"), ((True, 2), (False, 0)))
        def test_postprocessing_ic_stride(
            self, healthy_example_imu_data_ic_stride, healthy_example_stride_borders_ic_stride, var1, output
        ):
            """Test postprocessing for "ic" input_stride_type."""
            data_left = healthy_example_imu_data_ic_stride["left_sensor"]
            data_left.columns = BF_COLS
            # only use the first entry of the stride list
            stride_list_left = healthy_example_stride_borders_ic_stride["left_sensor"].iloc[0:1]

            def mock_func(event_list, *args, **kwargs):
                return event_list, None

            ed = self.algorithm_class(input_stride_type="ic", enforce_consistency=var1)
            with patch(
                "gaitmap._event_detection_common._event_detection_mixin.enforce_stride_list_consistency",
                side_effect=mock_func,
            ) as mock:
                ed.detect(data_left, stride_list_left, sampling_rate_hz=102.4)

            assert mock.call_count == output

    @pytest.mark.parametrize(("enforce_consistency", "output"), ((False, False), (True, True)))
    def test_disable_min_vel_event_list_segmented_stride(
        self, healthy_example_imu_data, healthy_example_stride_borders, enforce_consistency, output
    ):
        """Test for "segmented" input_stride_type."""
        data_left = healthy_example_imu_data["left_sensor"]
        data_left.columns = BF_COLS
        # only use the first entry of the stride list
        stride_list_left = healthy_example_stride_borders["left_sensor"].iloc[0:1]

        ed = self.algorithm_class(enforce_consistency=enforce_consistency)
        ed.detect(data_left, stride_list_left, sampling_rate_hz=204.8)

        assert hasattr(ed, "min_vel_event_list_") == output

    @pytest.mark.parametrize(("enforce_consistency", "output"), ((False, False), (True, True)))
    def test_disable_min_vel_event_list_ic_stride(
        self, healthy_example_imu_data_ic_stride, healthy_example_stride_borders_ic_stride, enforce_consistency, output
    ):
        """Test for "ic" input_stride_type."""
        data_left = healthy_example_imu_data_ic_stride["left_sensor"]
        data_left.columns = BF_COLS
        # only use the first entry of the stride list
        stride_list_left = healthy_example_stride_borders_ic_stride["left_sensor"].iloc[0:1]

        ed = self.algorithm_class(input_stride_type="ic", enforce_consistency=enforce_consistency)
        ed.detect(data_left, stride_list_left, sampling_rate_hz=102.4)

        assert hasattr(ed, "min_vel_event_list_") == output

    @common_arguments
    def test_multi_sensor_input_dict(self, input_stride_type, imu_data, stride_borders, sampling_rate, request):
        """Test to see if the algorithm is generally working on the example data when provided as dict."""
        data = coordinate_conversion.convert_to_fbf(
            request.getfixturevalue(imu_data), left=["left_sensor"], right=["right_sensor"]
        )

        dict_keys = ["l", "r"]
        data_dict = {dict_keys[0]: data["left_sensor"], dict_keys[1]: data["right_sensor"]}
        stride_list_dict = {
            dict_keys[0]: request.getfixturevalue(stride_borders)["left_sensor"],
            dict_keys[1]: request.getfixturevalue(stride_borders)["right_sensor"],
        }

        ed = self.algorithm_class(input_stride_type=input_stride_type)
        ed.detect(data_dict, stride_list_dict, sampling_rate_hz=sampling_rate)

        assert list(datatype_helper.get_multi_sensor_names(ed.min_vel_event_list_)) == dict_keys
        assert list(datatype_helper.get_multi_sensor_names(ed.segmented_event_list_)) == dict_keys

    @common_arguments
    def test_equal_output_dict_df(self, input_stride_type, imu_data, stride_borders, sampling_rate, request):
        """Test if output is similar for input dicts or regular multisensor data sets."""
        data = coordinate_conversion.convert_to_fbf(
            request.getfixturevalue(imu_data), left=["left_sensor"], right=["right_sensor"]
        )

        ed_df = self.algorithm_class(input_stride_type=input_stride_type)
        ed_df.detect(data, request.getfixturevalue(stride_borders), sampling_rate_hz=sampling_rate)

        dict_keys = ["l", "r"]
        data_dict = {dict_keys[0]: data["left_sensor"], dict_keys[1]: data["right_sensor"]}
        stride_list_dict = {
            dict_keys[0]: request.getfixturevalue(stride_borders)["left_sensor"],
            dict_keys[1]: request.getfixturevalue(stride_borders)["right_sensor"],
        }

        ed_dict = self.algorithm_class(input_stride_type=input_stride_type)
        ed_dict.detect(data_dict, stride_list_dict, sampling_rate_hz=sampling_rate)

        assert_frame_equal(ed_df.min_vel_event_list_["left_sensor"], ed_dict.min_vel_event_list_["l"])
        assert_frame_equal(ed_df.min_vel_event_list_["right_sensor"], ed_dict.min_vel_event_list_["r"])

    @pytest.mark.parametrize(
        ("input_stride_type", "stride_borders", "sampling_rate"),
        [
            ("segmented", "healthy_example_stride_borders", 204.8),
            ("ic", "healthy_example_stride_borders_ic_stride", 102.4),
        ],
    )
    def test_valid_input_data(self, input_stride_type, stride_borders, sampling_rate, request):
        """Test if error is raised correctly on invalid input data type."""
        data = pd.DataFrame({"a": [0, 1, 2], "b": [3, 4, 5]})
        ed = self.algorithm_class(input_stride_type=input_stride_type)
        with pytest.raises(ValidationError) as e:
            ed.detect(data, request.getfixturevalue(stride_borders), sampling_rate_hz=sampling_rate)

        assert "The passed object appears to be neither single- or multi-sensor data" in str(e)

    def test_min_vel_search_win_size_ms_dummy_data(self):
        """Test if error is raised correctly if windows size matches the size of the input data."""
        dummy_gyr = np.ones((100, 3))
        with pytest.raises(ValueError, match=r"min_vel_search_win_size_ms is*"):
            _detect_min_vel_gyr_energy(dummy_gyr, dummy_gyr.size)

    @common_arguments
    def test_valid_min_vel_search_win_size_ms(
        self, input_stride_type, imu_data, stride_borders, sampling_rate, request
    ):
        """Test if error is raised correctly on too large min_vel_search_win_size_ms."""
        data_left = request.getfixturevalue(imu_data)["left_sensor"]
        data_left = coordinate_conversion.convert_left_foot_to_fbf(data_left)
        stride_list_left = request.getfixturevalue(stride_borders)["left_sensor"]
        ed = self.algorithm_class(input_stride_type=input_stride_type, min_vel_search_win_size_ms=5000)
        with pytest.raises(ValueError, match=r"min_vel_search_win_size_ms is *"):
            ed.detect(data_left, stride_list_left, sampling_rate_hz=sampling_rate)

    @common_arguments
    def test_valid_ic_search_region_ms(self, input_stride_type, imu_data, stride_borders, sampling_rate, request):
        """Test if error is raised correctly on too small ic_search_region_ms."""
        data_left = request.getfixturevalue(imu_data)["left_sensor"]
        data_left = coordinate_conversion.convert_left_foot_to_fbf(data_left)
        stride_list_left = request.getfixturevalue(stride_borders)["left_sensor"]
        ed = self.algorithm_class(input_stride_type=input_stride_type, ic_search_region_ms=(1, 1))
        with pytest.raises(ValueError):
            ed.detect(data_left, stride_list_left, sampling_rate_hz=sampling_rate)

    @common_arguments
    def test_input_stride_list_size_one(self, input_stride_type, imu_data, stride_borders, sampling_rate, request):
        """Test if gait event detection also works with stride list of length 1."""
        data_left = request.getfixturevalue(imu_data)["left_sensor"]
        data_left = coordinate_conversion.convert_left_foot_to_fbf(data_left)
        # only use the first entry of the stride list
        stride_list_left = request.getfixturevalue(stride_borders)["left_sensor"].iloc[0:1]
        ed = self.algorithm_class(input_stride_type=input_stride_type)
        ed.detect(data_left, stride_list_left, sampling_rate_hz=sampling_rate)
        # per default min_vel_event_list_ has 6 columns
        assert_array_equal(np.array(ed.min_vel_event_list_.shape[1]), 6)
        # per default segmented_event_list_ has 5 columns
        assert_array_equal(np.array(ed.segmented_event_list_.shape[1]), 5)

    @common_arguments
    def test_correct_s_id(self, input_stride_type, imu_data, stride_borders, sampling_rate, request):
        """Test if the s_id from the stride list is correctly transferred to the output of event detection."""
        data_left = request.getfixturevalue(imu_data)["left_sensor"]
        data_left = coordinate_conversion.convert_left_foot_to_fbf(data_left)
        stride_list_left = request.getfixturevalue(stride_borders)["left_sensor"]
        # switch s_ids in stride list to random numbers
        stride_list_left["s_id"] = random.sample(range(1000), stride_list_left["s_id"].size)
        ed = self.algorithm_class(input_stride_type=input_stride_type)
        ed.detect(data_left, stride_list_left, sampling_rate_hz=sampling_rate)

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

    @common_arguments
    def test_single_data_multi_stride_list(self, input_stride_type, imu_data, stride_borders, sampling_rate, request):
        """Test correct error for combination of single sensor data set and multi sensor stride list."""
        data_left = request.getfixturevalue(imu_data)["left_sensor"]
        data_left = coordinate_conversion.convert_left_foot_to_fbf(data_left)
        stride_list_left = request.getfixturevalue(stride_borders)
        ed = self.algorithm_class(input_stride_type=input_stride_type)
        with pytest.raises(ValidationError):
            ed.detect(data_left, stride_list_left, sampling_rate_hz=sampling_rate)

    @common_arguments
    def test_multi_data_single_stride_list(self, input_stride_type, imu_data, stride_borders, sampling_rate, request):
        """Test correct error for combination of multi sensor data set and single sensor stride list."""
        data_left = request.getfixturevalue(imu_data)["left_sensor"]
        data_left = coordinate_conversion.convert_left_foot_to_fbf(data_left)
        stride_list_left = request.getfixturevalue(stride_borders)
        ed = self.algorithm_class(input_stride_type=input_stride_type)
        with pytest.raises(ValidationError):
            ed.detect(data_left, stride_list_left, sampling_rate_hz=sampling_rate)

    def test_sign_change_for_detect_tc_for_segmented_stride(self):
        """Test correct handling of signal that does or does not provide a change of the sign in case of "segmented" input_stride_type."""
        # with sign change
        signal1 = np.concatenate([np.ones(10), np.ones(10) * -1])
        assert _detect_tc_for_segmented_stride(signal1) == 9

        # without sign change
        signal2 = np.ones(10)
        assert np.isnan(_detect_tc_for_segmented_stride(signal2))

    def test_sign_change_for_detect_tc_for_ic_stride(self):
        """Test correct handling of signal that does or does not provide a change of the sign in case of "ic" input_stride_type."""
        # with sign change
        signal1 = np.concatenate([np.ones(10) * -1, np.ones(10), np.ones(10) * -1, np.ones(10)])
        assert _detect_tc_for_ic_stride(signal1) == 9

        # without sign change
        signal2 = np.ones(10)
        assert np.isnan(_detect_tc_for_ic_stride(signal2))

    @pytest.mark.parametrize(
        "detect_only",
        [
            ("min_vel",),
            ("ic",),
            ("tc",),
            ("min_vel", "ic"),
            ("ic", "tc"),
        ],
    )
    def test_detect_only_segmented_stride(self, detect_only, healthy_example_imu_data, healthy_example_stride_borders):
        """Test if only the specified events are detected."""
        data_left = healthy_example_imu_data["left_sensor"]
        data_left = coordinate_conversion.convert_left_foot_to_fbf(data_left)
        stride_list_left = healthy_example_stride_borders["left_sensor"]
        ed = RamppEventDetection(detect_only=detect_only)
        ed.detect(data_left, stride_list_left, sampling_rate_hz=204.8)

        expected_min_vel = ("pre_ic", *detect_only) if "ic" in detect_only else detect_only

        if "min_vel" in detect_only:
            assert set(ed.min_vel_event_list_.columns) == {"start", "end", *expected_min_vel}
        else:
            assert hasattr(ed, "min_vel_event_list_") is False

        assert set(ed.segmented_event_list_.columns) == {"start", "end", *detect_only}

    @pytest.mark.parametrize(
        "detect_only",
        [
            ("min_vel",),
            ("ic",),
            ("tc",),
            ("min_vel", "ic"),
            ("ic", "tc"),
        ],
    )
    def test_detect_only_ic_stride(
        self, detect_only, healthy_example_imu_data_ic_stride, healthy_example_stride_borders_ic_stride
    ):
        """Test if only the specified events are detected."""
        data_left = healthy_example_imu_data_ic_stride["left_sensor"]
        data_left = coordinate_conversion.convert_left_foot_to_fbf(data_left)
        stride_list_left = healthy_example_stride_borders_ic_stride["left_sensor"]
        ed = RamppEventDetection(input_stride_type="ic", detect_only=detect_only)
        ed.detect(data_left, stride_list_left, sampling_rate_hz=102.4)

        expected_min_vel = ("pre_ic", *detect_only) if "ic" in detect_only else detect_only

        if "min_vel" in detect_only:
            assert set(ed.min_vel_event_list_.columns) == {"start", "end", *expected_min_vel}
        else:
            assert hasattr(ed, "min_vel_event_list_") is False

        assert set(ed.segmented_event_list_.columns) == {"start", "end", *detect_only}
