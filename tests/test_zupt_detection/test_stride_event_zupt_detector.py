import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from gaitmap.utils.consts import SF_COLS
from gaitmap.utils.exceptions import ValidationError
from gaitmap.zupt_detection import StrideEventZuptDetector
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin


class TestMetaFunctionalityStrideEventZuptDetector(TestAlgorithmMixin):
    __test__ = True
    algorithm_class = StrideEventZuptDetector

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data):
        data_left = healthy_example_imu_data["left_sensor"].iloc[:10]
        return StrideEventZuptDetector().detect(
            data_left,
            sampling_rate_hz=204.8,
            stride_event_list=pd.DataFrame(
                [[0, 5, 0]], columns=["start", "end", "min_vel"], index=pd.Series([0], name="s_id")
            ),
        )


class TestStrideEventZuptDetector:
    def test_improper_stride_list(self):
        with pytest.raises(ValidationError):
            StrideEventZuptDetector().detect(
                pd.DataFrame([[0, 0, 0, 0, 0, 0]] * 10, columns=SF_COLS),
                sampling_rate_hz=1,
                # No min_vel column
                stride_event_list=pd.DataFrame([[0, 5]], columns=["start", "end"]),
            )

    def test_region_0(self):
        stride_event_list = pd.DataFrame(
            [[0, 7, 0], [5, 10, 5]], columns=["start", "end", "min_vel"], index=pd.Series([0, 1], name="s_id")
        )
        data = pd.DataFrame([[0, 0, 0, 0, 0, 0]] * 11, columns=SF_COLS)

        zupts = (
            StrideEventZuptDetector(half_region_size_s=0)
            .detect(data=data, stride_event_list=stride_event_list, sampling_rate_hz=1)
            .zupts_
        )

        assert_frame_equal(zupts, pd.DataFrame([[0, 1], [5, 6], [7, 8], [10, 11]], columns=["start", "end"]))

    def test_edge_case(self):
        """We test what happens if the zupt is exactly the first or last sample of the data or outside the range."""
        stride_event_list = pd.DataFrame(
            [[0, 10, 0], [10, 15, 10]], columns=["start", "end", "min_vel"], index=pd.Series([0, 1], name="s_id")
        )
        data = pd.DataFrame([[0, 0, 0, 0, 0, 0]] * 10, columns=SF_COLS)

        detector = StrideEventZuptDetector(half_region_size_s=0).detect(
            data=data, stride_event_list=stride_event_list, sampling_rate_hz=1
        )
        zupts = detector.zupts_

        assert_frame_equal(zupts, pd.DataFrame([[0, 1]], columns=["start", "end"]))
        assert detector.half_region_size_samples_ == 0

    def test_with_overlap(self):
        stride_event_list = pd.DataFrame(
            [[0, 7, 0], [5, 10, 5]], columns=["start", "end", "min_vel"], index=pd.Series([0, 1], name="s_id")
        )
        data = pd.DataFrame([[0, 0, 0, 0, 0, 0]] * 11, columns=SF_COLS)

        detector = StrideEventZuptDetector(half_region_size_s=2).detect(
            data=data, stride_event_list=stride_event_list, sampling_rate_hz=1
        )
        zupts = detector.zupts_
        assert_frame_equal(zupts, pd.DataFrame([[0, 11]], columns=["start", "end"]))
        assert detector.half_region_size_samples_ == 2

    def test_simple(self):
        stride_event_list = pd.DataFrame(
            [[0, 5, 0], [10, 15, 10]], columns=["start", "end", "min_vel"], index=pd.Series([0, 1], name="s_id")
        )
        data = pd.DataFrame([[0, 0, 0, 0, 0, 0]] * 20, columns=SF_COLS)

        detector = StrideEventZuptDetector(half_region_size_s=0.5).detect(
            data=data, stride_event_list=stride_event_list, sampling_rate_hz=2
        )
        zupts = detector.zupts_
        assert_frame_equal(zupts, pd.DataFrame([[0, 2], [4, 7], [9, 12], [14, 17]], columns=["start", "end"]))
        assert detector.half_region_size_samples_ == 1
