import numpy as np
import pandas as pd
import pytest

from gaitmap.base import BaseType
from gaitmap.trajectory_reconstruction.region_level_trajectory import RegionLevelTrajectory
from gaitmap.utils.consts import GF_POS
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin


# For further test see ./test_trajectory_wrapper.py


class TestMetaFunctionality(TestAlgorithmMixin):
    algorithm_class = RegionLevelTrajectory
    __test__ = True

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data, healthy_example_stride_events) -> BaseType:
        trajectory = RegionLevelTrajectory()
        trajectory.estimate(
            healthy_example_imu_data["left_sensor"],
            healthy_example_stride_events["left_sensor"].rename(columns={"s_id": "roi_id"}).iloc[:2],
            sampling_rate_hz=1,
        )
        return trajectory


class TestIntersect:
    @pytest.mark.parametrize("position, starts, ends", (
            ([-1, 0, 0, 0, 1, 1, 1, 2, 2, 2],  [0, 3, 6], [3, 6, 9]),
            ([-1, 0, 0, 0, -1, 1, 1, 1, 2, 2, 2], [0, 4, 7], [3, 7, 10]),
            ([-1, 0, 0, 0, 1, 1, 1, 2, 2, 2], [0], [9]),
    ))
    def test_intersect_single_region(self, position, starts, ends):
        # Note that the first value of position simulates the initial orientation calculated by the method.
        # The stride and roi list indices are relative to the data, which has one sample less.
        test_position = pd.DataFrame(
            np.array([[0] * len(position), position, position, position]).T,
            columns=("roi_id", *GF_POS),
        ).rename_axis("sample")
        test_roi_list = pd.DataFrame({"start": [0], "end": [len(position) - 1]}).rename_axis("roi_id")
        stride_list = pd.DataFrame({"start": starts, "end": ends}).rename_axis("s_id")
        rlt = RegionLevelTrajectory()
        rlt.position_ = test_position
        rlt.regions_of_interest = test_roi_list

        intersected_pos, *_ = rlt.intersect(stride_list, return_data=("position",))

        assert len(intersected_pos.groupby("s_id")) == len(starts)
        for i, (start, end) in enumerate(zip(starts, ends)):
            np.testing.assert_array_equal(intersected_pos["pos_x"].loc[i].to_numpy(), position[start: end + 1])


    def test_strides_outside_region(self):
        # Strides outside regions should simply be ignored
        position = [-1, 0, 0, 0, 1, 1, 1, 2, 2, 2]
        # 2 outside, 2 inside
        starts = [0, 3, 12, 18]
        ends = [3, 6, 15, 22]
        test_position = pd.DataFrame(
            np.array([[0] * len(position), position, position, position]).T,
            columns=("roi_id", *GF_POS),
        ).rename_axis("sample")
        test_roi_list = pd.DataFrame({"start": [0], "end": [len(position) - 1]}).rename_axis("roi_id")
        stride_list = pd.DataFrame({"start": starts, "end": ends}).rename_axis("s_id")
        rlt = RegionLevelTrajectory()
        rlt.position_ = test_position
        rlt.regions_of_interest = test_roi_list

        intersected_pos, *_ = rlt.intersect(stride_list, return_data=("position",))

        assert len(intersected_pos.groupby("s_id")) == 2


