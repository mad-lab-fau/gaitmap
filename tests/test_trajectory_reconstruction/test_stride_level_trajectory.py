import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas._testing import assert_frame_equal
from scipy.spatial.transform import Rotation

from gaitmap.base import BaseType
from gaitmap.trajectory_reconstruction.stride_level_trajectory import StrideLevelTrajectory
from gaitmap.utils.consts import SF_COLS
from gaitmap.utils.dataset_helper import (
    is_single_sensor_orientation_list,
    is_single_sensor_position_list,
    is_multi_sensor_orientation_list,
    is_multi_sensor_position_list,
)
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


class TestIODataStructures:
    def test_invalid_input_data(self, healthy_example_imu_data, healthy_example_stride_events):
        """Test if error is raised correctly on invalid input data type"""
        data = healthy_example_imu_data
        stride_list = healthy_example_stride_events
        fake_data = pd.DataFrame({"a": [0, 1, 2], "b": [3, 4, 5]})
        fake_stride_list = {
            "a": pd.DataFrame(data=[[0, 1, 2]], columns=["stride", "begin", "stop"]),
            "b": pd.DataFrame(data=[[0, 1, 2]], columns=["stride", "begin", "stop"]),
        }

        gyr_int = StrideLevelTrajectory(align_window_width=8)
        error_text = r"Provided data or stride list or combination of both is not supported by gaitmap."
        with pytest.raises(ValueError, match=error_text):
            gyr_int.estimate(fake_data, stride_list, 204.8)
        with pytest.raises(ValueError, match=error_text):
            gyr_int.estimate(data, fake_stride_list, 204.8)

    def test_invalid_input_method(self, healthy_example_imu_data, healthy_example_stride_events):
        """Test if correct errors are raised for invalid pos and ori methods."""
        instance = StrideLevelTrajectory(ori_method="wrong")
        with pytest.raises(ValueError) as e:
            instance.estimate(healthy_example_imu_data, healthy_example_stride_events, 204.8)
        assert "ori_method" in str(e)

        instance = StrideLevelTrajectory(pos_method="wrong")
        with pytest.raises(ValueError) as e:
            instance.estimate(healthy_example_imu_data, healthy_example_stride_events, 204.8)
        assert "pos_method" in str(e)

    def test_single_sensor_output(self, healthy_example_imu_data, healthy_example_stride_events, snapshot):
        test_stride_events = healthy_example_stride_events["left_sensor"].iloc[:3]
        test_data = healthy_example_imu_data["left_sensor"].iloc[: int(test_stride_events.iloc[-1]["end"])]

        instance = StrideLevelTrajectory()
        instance.estimate(test_data, test_stride_events, 204.8)

        assert is_single_sensor_orientation_list(instance.orientation_)
        assert is_single_sensor_position_list(instance.position_)
        assert_array_equal(instance.orientation_.reset_index()["s_id"].unique(), test_stride_events["s_id"].unique())
        assert_array_equal(instance.position_.reset_index()["s_id"].unique(), test_stride_events["s_id"].unique())
        assert_array_equal(instance.velocity_.reset_index()["s_id"].unique(), test_stride_events["s_id"].unique())

        for _, s in test_stride_events.iterrows():
            assert len(instance.orientation_.loc[s["s_id"]]) == s["end"] - s["start"] + 1
            assert len(instance.position_.loc[s["s_id"]]) == s["end"] - s["start"] + 1
            assert len(instance.velocity_.loc[s["s_id"]]) == s["end"] - s["start"] + 1

        first_last_stride = test_stride_events.iloc[[0, -1]]["s_id"]
        snapshot.assert_match(instance.orientation_.loc[first_last_stride], "ori")
        snapshot.assert_match(instance.position_.loc[first_last_stride], "pos")

    def test_multi_sensor_output(self, healthy_example_imu_data, healthy_example_stride_events, snapshot):
        test_stride_events = healthy_example_stride_events
        test_data = healthy_example_imu_data

        instance = StrideLevelTrajectory()
        instance.estimate(test_data, test_stride_events, 204.8)

        assert is_multi_sensor_orientation_list(instance.orientation_)
        assert is_multi_sensor_position_list(instance.position_)
        for sensor in test_stride_events.keys():
            assert_array_equal(
                instance.orientation_[sensor].reset_index()["s_id"].unique(),
                test_stride_events[sensor]["s_id"].unique(),
            )
            assert_array_equal(
                instance.position_[sensor].reset_index()["s_id"].unique(), test_stride_events[sensor]["s_id"].unique()
            )
            assert_array_equal(
                instance.velocity_[sensor].reset_index()["s_id"].unique(), test_stride_events[sensor]["s_id"].unique()
            )

            for _, s in test_stride_events[sensor].iterrows():
                assert len(instance.orientation_[sensor].loc[s["s_id"]]) == s["end"] - s["start"] + 1
                assert len(instance.position_[sensor].loc[s["s_id"]]) == s["end"] - s["start"] + 1
                assert len(instance.velocity_[sensor].loc[s["s_id"]]) == s["end"] - s["start"] + 1

            first_last_stride = test_stride_events[sensor].iloc[[0, -1]]["s_id"]
            snapshot.assert_match(instance.orientation_[sensor].loc[first_last_stride], "ori_{}".format(sensor))
            snapshot.assert_match(
                instance.position_[sensor].loc[first_last_stride], "pos_{}".format(sensor),
            )


class TestInitCalculation:
    """Test the calculation of initial rotations per stride.

    No complicated tests here, as this uses `get_gravity_rotation`, which is well tested
    """

    def test_calc_initial_dummy(self):
        """No rotation expected as already aligned."""
        dummy_data = pd.DataFrame(np.repeat(np.array([0, 0, 1, 0, 0, 0])[None, :], 20, axis=0), columns=SF_COLS)
        start_ori = StrideLevelTrajectory.calculate_initial_orientation(dummy_data, 10, 8)
        assert_array_equal(start_ori.as_quat(), Rotation.identity().as_quat())

    @pytest.mark.parametrize("start", [0, 99])
    def test_start_of_stride_equals_start_or_end_of_data(self, start):
        """If start is to close to the start or the end of the data a warning is emitted."""
        dummy_data = pd.DataFrame(np.repeat(np.array([0, 0, 1, 0, 0, 0])[None, :], 100, axis=0), columns=SF_COLS)
        with pytest.warns(UserWarning) as w:
            StrideLevelTrajectory.calculate_initial_orientation(dummy_data, start, 8)

        assert "complete window length" in str(w[0])
