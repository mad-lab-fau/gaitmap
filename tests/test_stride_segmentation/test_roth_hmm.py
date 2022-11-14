import numpy as np
import pandas as pd
import pytest

from gaitmap.utils.coordinate_conversion import convert_left_foot_to_fbf
from gaitmap_mad.stride_segmentation.hmm import PreTrainedRothSegmentationModel, RothHMM, SimpleSegmentationHMM
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin


class TestMetaFunctionalitySegmentationModel(TestAlgorithmMixin):
    __test__ = True

    algorithm_class = SimpleSegmentationHMM

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data) -> SimpleSegmentationHMM:
        hmm = PreTrainedRothSegmentationModel()
        hmm.predict(convert_left_foot_to_fbf(healthy_example_imu_data["left_sensor"]), sampling_rate_hz=100)
        return hmm


class TestMetaFunctionalityRothHmm(TestAlgorithmMixin):
    __test__ = True

    algorithm_class = RothHMM

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data) -> RothHMM:
        hmm = RothHMM()
        hmm.segment(convert_left_foot_to_fbf(healthy_example_imu_data["left_sensor"]), sampling_rate_hz=100)
        return hmm
