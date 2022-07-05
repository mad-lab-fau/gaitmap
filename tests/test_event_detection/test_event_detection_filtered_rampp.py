import pytest

from gaitmap.base import BaseType
from gaitmap.utils.consts import BF_COLS
from gaitmap_mad.event_detection import FilteredRamppEventDetection
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin
from tests.mixins.test_caching_mixin import TestCachingMixin


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