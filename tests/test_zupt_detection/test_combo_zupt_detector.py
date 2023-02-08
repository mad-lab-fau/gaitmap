from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from gaitmap.base import BaseZuptDetector
from gaitmap.utils.consts import SF_COLS
from gaitmap.zupt_detection import ComboZuptDetector, NormZuptDetector, RegionZuptDetectorMixin
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin


class TestMetaFunctionalityComboZuptDetector(TestAlgorithmMixin):
    __test__ = True
    algorithm_class = ComboZuptDetector

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data):
        data_left = healthy_example_imu_data["left_sensor"].iloc[:500]
        return ComboZuptDetector([("a", NormZuptDetector()), ("b", NormZuptDetector())]).detect(
            data_left, sampling_rate_hz=204.8
        )


class DummyZuptDetector(BaseZuptDetector, RegionZuptDetectorMixin):
    def __init__(self, zupts):
        self.zupts = zupts

    def detect(self, data, sampling_rate_hz, **kwargs):
        self.data = data
        self.zupts_ = self.zupts
        return self


class TestComboZuptDetector:
    @pytest.mark.parametrize("detector_list", [None, []])
    def test_empty_detector_list(self, detector_list):
        with pytest.raises(ValueError):
            ComboZuptDetector(detector_list).detect(pd.DataFrame(), sampling_rate_hz=1)

    def test_kwargs_forwarded(self):
        class MockZUPTDetector(BaseZuptDetector):
            zupts_ = pd.DataFrame(columns=["start", "end"])
            per_sample_zupts_ = np.zeros(10)

        with patch.object(MockZUPTDetector, "detect") as mock_detect:
            mock_detect.return_value = MockZUPTDetector()
            test = ComboZuptDetector([("a", MockZUPTDetector()), ("b", MockZUPTDetector())])
            test.detect(pd.DataFrame(np.zeros((10, 6)), columns=SF_COLS), sampling_rate_hz=1, foo="bar")

            assert mock_detect.call_count == 2
            for call in mock_detect.call_args_list:
                assert call.kwargs["foo"] == "bar"

    def test_dummy(self):
        zupts = pd.DataFrame(
            [[0, 10], [30, 55], [85, 90]],
            columns=["start", "end"],
        )
        test = ComboZuptDetector([("a", DummyZuptDetector(zupts))])
        test.detect(pd.DataFrame(np.zeros(100)), sampling_rate_hz=1)
        assert_array_equal(test.zupts_, zupts)

    def test_empty_data_edge_case(self):
        zupts = pd.DataFrame(
            [[0, 10], [30, 55], [85, 90]],
            columns=["start", "end"],
        )
        test = ComboZuptDetector([("a", DummyZuptDetector(zupts))])
        test.detect(pd.DataFrame(), sampling_rate_hz=1)
        assert_array_equal(test.zupts_, pd.DataFrame(columns=["start", "end"]))

    def test_combine_with_or(self):
        zupts_a = pd.DataFrame(
            [[0, 10], [30, 55], [85, 90]],
            columns=["start", "end"],
        )
        zupts_b = pd.DataFrame(
            [[0, 10], [35, 60], [95, 100]],
            columns=["start", "end"],
        )
        test = ComboZuptDetector([("a", DummyZuptDetector(zupts_a)), ("b", DummyZuptDetector(zupts_b))], operation="or")
        test.detect(pd.DataFrame(np.zeros(100)), sampling_rate_hz=1)
        assert_array_equal(
            test.zupts_, pd.DataFrame([[0, 10], [30, 60], [85, 90], [95, 100]], columns=["start", "end"])
        )

    def test_combine_with_and(self):
        zupts_a = pd.DataFrame(
            [[0, 10], [30, 55], [85, 90]],
            columns=["start", "end"],
        )
        zupts_b = pd.DataFrame(
            [[0, 10], [35, 60], [95, 100]],
            columns=["start", "end"],
        )
        test = ComboZuptDetector(
            [("a", DummyZuptDetector(zupts_a)), ("b", DummyZuptDetector(zupts_b))], operation="and"
        )
        test.detect(pd.DataFrame(np.zeros(100)), sampling_rate_hz=1)
        assert_array_equal(test.zupts_, pd.DataFrame([[0, 10], [35, 55]], columns=["start", "end"]))
