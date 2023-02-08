from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from scipy.spatial.transform import Rotation

from gaitmap.base import BaseTrajectoryMethod, BaseType
from gaitmap.trajectory_reconstruction._stride_level_trajectory import StrideLevelTrajectory
from gaitmap.utils.consts import GF_POS, GF_VEL, SF_COLS
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin


class TestMetaFunctionality(TestAlgorithmMixin):
    algorithm_class = StrideLevelTrajectory
    __test__ = True

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data, healthy_example_stride_events) -> BaseType:
        trajectory = StrideLevelTrajectory()
        trajectory.estimate(
            healthy_example_imu_data["left_sensor"],
            healthy_example_stride_events["left_sensor"].iloc[:2],
            sampling_rate_hz=1,
        )
        return trajectory


class MockTrajectory(BaseTrajectoryMethod):
    def __init__(self, initial_orientation=None):
        self.initial_orientation = initial_orientation
        super().__init__()

    orientation_object_ = Rotation.from_quat(np.array([[0, 0, 0, 1]] * 10))
    velocity_ = pd.DataFrame(np.zeros((10, 3)), columns=GF_VEL)
    position_ = pd.DataFrame(np.zeros((10, 3)), columns=GF_POS)


class TestStrideLevelTrajectory:
    def test_event_list_forwarded(self):
        with patch.object(MockTrajectory, "estimate") as mock_estimate:
            mock_estimate.return_value = MockTrajectory()
            test = StrideLevelTrajectory(ori_method=None, pos_method=None, trajectory_method=MockTrajectory())

            test.estimate(
                pd.DataFrame(np.array([[0, 0, 9.81, 0, 0, 0]] * 10), columns=SF_COLS),
                stride_event_list=pd.DataFrame(
                    {"start": [0, 5], "end": [5, 12], "min_vel": [0, 5]}, index=pd.Series([2, 3], name="s_id")
                ),
                sampling_rate_hz=1,
            )

            # Mock should be called twice, once for each stride
            assert mock_estimate.call_count == 2
            # First call should be for the first stride
            assert_frame_equal(
                mock_estimate.call_args_list[0].kwargs["stride_event_list"],
                pd.DataFrame({"start": [0], "end": [5], "min_vel": [0]}, index=pd.Series([2], name="s_id")),
            )
            # Second call should be for the second stride
            # However, we normalize each stride (i.e. start at 0)
            assert_frame_equal(
                mock_estimate.call_args_list[1].kwargs["stride_event_list"],
                pd.DataFrame({"start": [0], "end": [7], "min_vel": [0]}, index=pd.Series([3], name="s_id")),
            )

    def test_event_list_forwarded_multi(self):
        with patch.object(MockTrajectory, "estimate") as mock_estimate:
            mock_estimate.return_value = MockTrajectory()
            test = StrideLevelTrajectory(ori_method=None, pos_method=None, trajectory_method=MockTrajectory())

            test.estimate(
                {
                    "left_sensor": pd.DataFrame(np.array([[0, 0, 9.81, 0, 0, 0]] * 10), columns=SF_COLS),
                    "right_sensor": pd.DataFrame(np.array([[0, 0, 9.81, 0, 0, 0]] * 10), columns=SF_COLS),
                },
                stride_event_list={
                    "left_sensor": pd.DataFrame(
                        {"start": [0, 5], "end": [5, 12], "min_vel": [0, 5]},
                        index=pd.Series(["l_2", "l_3"], name="s_id"),
                    ),
                    "right_sensor": pd.DataFrame(
                        {"start": [0, 5], "end": [5, 12], "min_vel": [0, 5]},
                        index=pd.Series(["r_2", "r_3"], name="s_id"),
                    ),
                },
                sampling_rate_hz=1,
            )

            # Mock should be called twice, once for each stride
            assert mock_estimate.call_count == 4
            # First call should be for the first stride
            calls = iter(mock_estimate.call_args_list)
            for sensor in ["left_sensor", "right_sensor"]:
                assert_frame_equal(
                    next(calls).kwargs["stride_event_list"],
                    pd.DataFrame(
                        {"start": [0], "end": [5], "min_vel": [0]}, index=pd.Series([f"{sensor[0]}_2"], name="s_id")
                    ),
                )
                assert_frame_equal(
                    next(calls).kwargs["stride_event_list"],
                    pd.DataFrame(
                        {"start": [0], "end": [7], "min_vel": [0]}, index=pd.Series([f"{sensor[0]}_3"], name="s_id")
                    ),
                )
