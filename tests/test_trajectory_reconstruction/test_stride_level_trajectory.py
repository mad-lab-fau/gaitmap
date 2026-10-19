from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gaitmap.base import BaseType
from gaitmap.trajectory_reconstruction._stride_level_trajectory import StrideLevelTrajectory
from gaitmap.utils.consts import SF_COLS
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin
from tests.test_trajectory_reconstruction.test_trajectory_wrapper import MockTrajectory


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


class TestStrideLevelTrajectory:
    def test_event_list_forwarded(self) -> None:
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

    def test_event_list_forwarded_multi(self) -> None:
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


@pytest.mark.parametrize("multi_sensor", [False, True])
def test_event_based_reconstruction_accepts_float_stride_metadata(multi_sensor):
    """Metadata must not change the sample coordinates forwarded to event ZUPTs."""
    from gaitmap.trajectory_reconstruction import RtsKalman
    from gaitmap.zupt_detection import StrideEventZuptDetector

    data = pd.DataFrame(0.0, index=range(80), columns=SF_COLS)
    data["acc_z"] = 9.81
    strides = pd.DataFrame({"start": [10, 40], "end": [30, 60], "min_vel": [10, 40]}).rename_axis("s_id")
    with_metadata = strides.assign(confidence=[0.5, 0.75])
    if multi_sensor:
        data, strides, with_metadata = {"foot": data}, {"foot": strides}, {"foot": with_metadata}
    reconstruction = StrideLevelTrajectory(
        ori_method=None, pos_method=None, trajectory_method=RtsKalman(zupt_detector=StrideEventZuptDetector())
    )
    expected = reconstruction.clone().estimate(data, strides, sampling_rate_hz=100)

    actual = reconstruction.estimate(data, with_metadata, sampling_rate_hz=100)

    for attribute in ["position_", "velocity_", "orientation_"]:
        result, reference = getattr(actual, attribute), getattr(expected, attribute)
        if multi_sensor:
            result, reference = result["foot"], reference["foot"]
        assert_frame_equal(result, reference)
        assert not result.empty
