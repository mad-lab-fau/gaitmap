import pytest
from pandas._testing import assert_frame_equal

from gaitmap.base import BaseType
from gaitmap.event_detection import FilteredRamppEventDetection
from gaitmap.utils import coordinate_conversion
from gaitmap.utils.consts import BF_COLS
from gaitmap.event_detection import RamppEventDetection
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin
from tests.mixins.test_caching_mixin import TestCachingMixin
from tests.test_event_detection.test_event_detection_rampp import TestEventDetectionRampp


class MetaTestConfig:
    algorithm_class = FilteredRamppEventDetection

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data, healthy_example_stride_borders) -> BaseType:
        data_left = healthy_example_imu_data["left_sensor"]
        data_left.columns = BF_COLS
        # only use the first entry of the stride list
        stride_list_left = healthy_example_stride_borders["left_sensor"].iloc[0:1]
        ed = FilteredRamppEventDetection()
        ed.detect(data_left, stride_list_left, 204.8)
        return ed


class TestMetaFunctionality(MetaTestConfig, TestAlgorithmMixin):
    __test__ = True


class TestCachingFunctionality(MetaTestConfig, TestCachingMixin):
    __test__ = True


class TestEventDetectionRamppFiltered(TestEventDetectionRampp):
    algorithm_class = FilteredRamppEventDetection

    def test_is_identical_to_normal_rampp(self, healthy_example_imu_data, healthy_example_stride_borders, snapshot):
        """Test if the output is the same as normal Rampp for lax filter parameters."""
        data = coordinate_conversion.convert_to_fbf(
            healthy_example_imu_data, left=["left_sensor"], right=["right_sensor"]
        )

        ed = self.algorithm_class()
        ed.detect(data, healthy_example_stride_borders, 204.8)
        rampp_ed = RamppEventDetection()
        rampp_ed.detect(data, healthy_example_stride_borders, 204.8)

        for sensor in ("left_sensor", "right_sensor"):
            assert_frame_equal(ed.segmented_event_list_[sensor], rampp_ed.segmented_event_list_[sensor])

    @pytest.mark.parametrize("filter_paras", [(3, 5), (2, 10)])
    def test_correct_arguments_are_passed(self, healthy_example_imu_data, healthy_example_stride_borders, filter_paras):
        data = coordinate_conversion.convert_to_fbf(
            healthy_example_imu_data, left=["left_sensor"], right=["right_sensor"]
        )

        ed = self.algorithm_class(ic_lowpass_filter_parameter=filter_paras)
        ed.detect(data, healthy_example_stride_borders, 204.8)

        assert ed._get_detect_kwargs()["gyr_ic_lowpass_filter_parameters"] == filter_paras
        assert ed._get_detect_kwargs()["sampling_rate_hz"] == 204.8
