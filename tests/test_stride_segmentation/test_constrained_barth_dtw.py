"""Tests for ConstrainedBarthDtw.

We only test the meta functionality and a regressions, as it is basically identical to BarthDtw, except some defaults.
"""
import numpy as np
import pytest

from gaitmap.base import BaseType
from gaitmap.stride_segmentation import ConstrainedBarthDtw, create_dtw_template, BarthOriginalTemplate
from gaitmap.utils.coordinate_conversion import convert_to_fbf
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin


class TestMetaFunctionality(TestAlgorithmMixin):
    algorithm_class = ConstrainedBarthDtw
    __test__ = True

    @pytest.fixture()
    def after_action_instance(self) -> BaseType:
        template = create_dtw_template(np.array([0, 1.0, 0]), sampling_rate_hz=100.0)
        dtw = self.algorithm_class(template=template, max_cost=0.5, min_match_length_s=None)
        data = np.array([0, 1.0, 0])
        dtw.segment(data, sampling_rate_hz=100)
        return dtw


class TestRegressionOnRealDataConstrainedDtw:
    """These regression tests run on a MS dataset, which produces a bunch of issues wiht the normal dtw."""
    def test_real_data_both_feed_regression(self, ms_example_imu_data, snapshot):
        data = convert_to_fbf(ms_example_imu_data, right=["right_sensor"], left=["left_sensor"])
        dtw = ConstrainedBarthDtw(template=BarthOriginalTemplate(use_cols=("gyr_ml", "gyr_si")))  # Test with default
        # paras
        dtw.segment(data, sampling_rate_hz=102.4)

        snapshot.assert_match(dtw.stride_list_["left_sensor"], "left")
        snapshot.assert_match(dtw.stride_list_["right_sensor"], "right")
