import numpy as np
import pandas as pd
import pytest

from gaitmap.base import BaseType
from gaitmap.trajectory_reconstruction.region_level_trajectory import RegionLevelTrajectory
from gaitmap.utils.consts import GF_POS, SF_COLS
from gaitmap.utils.datatype_helper import (
    is_single_sensor_position_list,
    is_single_sensor_orientation_list,
    is_single_sensor_velocity_list,
    is_multi_sensor_position_list,
)
from gaitmap.utils.exceptions import ValidationError
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

    def test_all_other_parameters_documented(self):
        # As the class has multiple action methods with different parameters, this test can not pass in its current
        # state
        pytest.skip()


class TestEstimateIntersect:
    @pytest.mark.parametrize("sl_type, roi_type", (("single", "multi"), ("multi", "single")))
    def test_datatypes_mismatch(self, sl_type, roi_type):
        roi_list = pd.DataFrame({"start": [0], "end": [8]}).rename_axis("roi_id")
        stride_list = pd.DataFrame({"start": [0], "end": [1]}).rename_axis("s_id")
        if roi_type == "multi":
            roi_list = {"s1": roi_list}
        if sl_type == "multi":
            stride_list = {"s1": stride_list}

        rlt = RegionLevelTrajectory()

        with pytest.raises(ValidationError) as e:
            rlt.estimate_intersect({}, roi_list, stride_list, sampling_rate_hz=1)

        assert "The stride list is {} sensor and the stride list is {} sensor.".format(sl_type, roi_type) in str(e)

    def test_simple(self):
        acc_xy = pd.Series([0, 1, 0, -1, 0, 0, 1, 0, -1, 0, 0, 1, 0, -1, 0, 0])
        acc_z = pd.Series([9.81] * len(acc_xy))
        gyr = pd.Series([0.0] * len(acc_xy))
        data = pd.concat([acc_xy, acc_xy, acc_z, gyr, gyr, gyr], axis=1)
        data.columns = SF_COLS
        starts, ends = [0, 5, 10], [5, 10, 15]
        stride_list = pd.DataFrame({"start": starts, "end": ends}).rename_axis("s_id")
        roi_list = pd.DataFrame({"start": [0], "end": [len(data)]}).rename_axis("roi_id")

        rlt = RegionLevelTrajectory(align_window_width=0)
        rlt.estimate_intersect(data, roi_list, stride_list, sampling_rate_hz=1)

        # Output should pass stride pos list test
        assert is_single_sensor_position_list(rlt.position_, "stride")
        assert is_single_sensor_orientation_list(rlt.orientation_, "stride")
        assert is_single_sensor_velocity_list(rlt.velocity_, "stride")

        assert len(rlt.position_.groupby("s_id")) == len(starts)
        assert len(rlt.position_) == len(starts) * 6

        for _, s in rlt.velocity_.groupby("s_id"):
            np.testing.assert_array_equal(s["vel_x"].to_numpy(), [0, 0, 0.5, 1, 0.5, 0])


class TestIntersect:
    @pytest.mark.parametrize(
        "position, starts, ends",
        (
            ([-1, 0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 3, 6], [3, 6, 9]),
            ([-1, 0, 0, 0, -1, 1, 1, 1, 2, 2, 2], [0, 4, 7], [3, 7, 10]),
            ([-1, 0, 0, 0, 1, 1, 1, 2, 2, 2], [0], [9]),
        ),
    )
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
            np.testing.assert_array_equal(intersected_pos["pos_x"].loc[i].to_numpy(), position[start : end + 1])

        # Output should pass stride pos list test
        assert is_single_sensor_position_list(intersected_pos, "stride")

    def test_strides_outside_region(self):
        # Strides outside regions should simply be ignored
        position = [-1, 0, 0, 0, 1, 1, 1, 2, 2, 2]
        # 2 outside, 2 inside
        starts = [0, 3, 12, 18]
        ends = [3, 6, 15, 22]
        test_position = pd.DataFrame(
            np.array([[0] * len(position), position, position, position]).T, columns=("roi_id", *GF_POS)
        ).rename_axis("sample")
        test_roi_list = pd.DataFrame({"start": [0], "end": [len(position) - 1]}).rename_axis("roi_id")
        stride_list = pd.DataFrame({"start": starts, "end": ends}).rename_axis("s_id")
        rlt = RegionLevelTrajectory()
        rlt.position_ = test_position
        rlt.regions_of_interest = test_roi_list

        intersected_pos, *_ = rlt.intersect(stride_list, return_data=("position",))

        assert len(intersected_pos.groupby("s_id")) == 2

    def test_multiple_roi(self):
        # two gait sequences with some padding in between
        position = {"gs1": pd.Series([-1, 0, 0, 0, 1, 1, 1]), "gs2": pd.Series([-1, 2, 2, 2, 3, 3, 3])}
        test_roi_list = pd.DataFrame({"roi_id": ["gs1", "gs2"], "start": [0, 8], "end": [6, 14]})
        # 2 per gait sequences
        starts = [0, 3, 8, 11]
        ends = [3, 6, 11, 14]
        test_position = pd.concat(position)
        test_position = pd.concat([test_position, test_position, test_position], axis=1).rename_axis(
            ["roi_id", "sample"]
        )
        test_position.columns = GF_POS
        stride_list = pd.DataFrame({"start": starts, "end": ends}).rename_axis("s_id")
        rlt = RegionLevelTrajectory()
        rlt.position_ = test_position
        rlt.regions_of_interest = test_roi_list

        intersected_pos, *_ = rlt.intersect(stride_list, return_data=("position",))

        assert len(intersected_pos.groupby("s_id")) == len(starts)
        np.testing.assert_array_equal(intersected_pos["pos_x"].loc[0].to_numpy(), [-1, 0, 0, 0])
        np.testing.assert_array_equal(intersected_pos["pos_x"].loc[1].to_numpy(), [0, 1, 1, 1])
        np.testing.assert_array_equal(intersected_pos["pos_x"].loc[2].to_numpy(), [-1, 2, 2, 2])
        np.testing.assert_array_equal(intersected_pos["pos_x"].loc[3].to_numpy(), [2, 3, 3, 3])

    def test_overlapping_roi(self):
        # Overlapping rois the stride information should be taken from the last roi.
        position = {"gs1": pd.Series([-1, 0, 0, 0, 1, 1, 1]), "gs2": pd.Series([-1, 2, 2, 2, 3, 3, 3])}
        # ROIs completely overlap
        test_roi_list = pd.DataFrame({"roi_id": ["gs1", "gs2"], "start": [0, 0], "end": [8, 8]})
        starts = [0, 3]
        ends = [3, 6]
        test_position = pd.concat(position)
        test_position = pd.concat([test_position, test_position, test_position], axis=1).rename_axis(
            ["roi_id", "sample"]
        )
        test_position.columns = GF_POS
        stride_list = pd.DataFrame({"start": starts, "end": ends}).rename_axis("s_id")
        rlt = RegionLevelTrajectory()
        rlt.position_ = test_position
        rlt.regions_of_interest = test_roi_list

        intersected_pos, *_ = rlt.intersect(stride_list, return_data=("position",))

        np.testing.assert_array_equal(intersected_pos["pos_x"].loc[0].to_numpy(), [-1, 2, 2, 2])
        np.testing.assert_array_equal(intersected_pos["pos_x"].loc[1].to_numpy(), [2, 3, 3, 3])

    def test_estimate_must_be_called(self):
        with pytest.raises(ValidationError) as e:
            RegionLevelTrajectory().intersect({})

        assert "`estimate`" in str(e)

    def test_estimate_intersect_was_called(self):
        # Simulate calling `estimate_intersect` by setting a stride-level pos list:
        position = [1, 1, 1]
        position_list = pd.DataFrame(
            np.array([[0] * len(position), position, position, position]).T,
            columns=("s_id", *GF_POS),
        ).rename_axis("sample")

        stride_list = pd.DataFrame({"start": [0], "end": [1]}).rename_axis("s_id")
        rlt = RegionLevelTrajectory()
        rlt.position_ = position_list

        with pytest.raises(ValidationError) as e:
            rlt.intersect(stride_list, return_data=("position",))

        assert "`estimate_intersect`" in str(e)

    @pytest.mark.parametrize("sl_type, pos_type", (("single", "multi"), ("multi", "single")))
    def test_datatypes_mismatch(self, sl_type, pos_type):
        position = [1, 1, 1]
        position_list = pd.DataFrame(
            np.array([[0] * len(position), position, position, position]).T,
            columns=("roi_id", *GF_POS),
        ).rename_axis("sample")
        roi_list = pd.DataFrame({"start": [0], "end": [len(position) - 1]}).rename_axis("roi_id")
        stride_list = pd.DataFrame({"start": [0], "end": [1]}).rename_axis("s_id")
        if pos_type == "multi":
            position_list = {"s1": position_list}
            roi_list = {"s1": roi_list}
        if sl_type == "multi":
            stride_list = {"s1": stride_list}

        rlt = RegionLevelTrajectory()
        rlt.position_ = position_list
        rlt.regions_of_interest = roi_list

        with pytest.raises(ValidationError) as e:
            rlt.intersect(stride_list, return_data=("position",))

        assert "{} sensor dataset with a {} sensor stride list".format(pos_type, sl_type) in str(e)

    @pytest.mark.parametrize("value", (tuple(), "invalid", ("invalid1", "orientation")))
    def test_invalid_return_data(self, value):
        rlt = RegionLevelTrajectory()
        rlt.position_ = [1]  # Something other than None
        with pytest.raises(ValueError) as e:
            rlt.intersect({}, value)
        assert str(("orientation", "position", "velocity")) in str(e)

    def test_data_has_been_modified(self):
        rlt = RegionLevelTrajectory()
        # Simulate non-valid result properties
        rlt.position_ = "not a valid position list"

        with pytest.raises(ValueError) as e:
            rlt.intersect(pd.DataFrame({"start": [0], "end": [1]}).rename_axis("s_id"), ("position",))

        assert "manipulated the outputs" in str(e)

    def test_multiple_sensors(self):
        position = [-1, 0, 0, 0, 1, 1, 1]
        starts = [0, 3]
        ends = [3, 6]
        test_position = pd.DataFrame(
            np.array([[0] * len(position), position, position, position]).T,
            columns=("roi_id", *GF_POS),
        ).rename_axis("sample")
        test_roi_list = pd.DataFrame({"start": [0], "end": [len(position) - 1]}).rename_axis("roi_id")
        stride_list = pd.DataFrame({"start": starts, "end": ends}).rename_axis("s_id")
        rlt = RegionLevelTrajectory()
        rlt.position_ = {"s1": test_position, "s2": test_position}
        rlt.regions_of_interest = {"s1": test_roi_list, "s2": test_roi_list}

        intersected_pos, *_ = rlt.intersect({"s1": stride_list, "s2": stride_list}, return_data=("position",))

        assert len(intersected_pos["s1"].groupby("s_id")) == 2
        assert len(intersected_pos["s2"].groupby("s_id")) == 2

        # Output should pass stride pos list test
        assert is_multi_sensor_position_list(intersected_pos, "stride")
