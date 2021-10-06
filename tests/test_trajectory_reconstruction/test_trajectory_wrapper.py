from typing import Dict, Type

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from scipy.spatial.transform import Rotation
from typing_extensions import Literal

from gaitmap.base import BaseTrajectoryReconstructionWrapper
from gaitmap.trajectory_reconstruction import RegionLevelTrajectory, RtsKalman
from gaitmap.trajectory_reconstruction._trajectory_wrapper import _initial_orientation_from_start
from gaitmap.trajectory_reconstruction.stride_level_trajectory import StrideLevelTrajectory
from gaitmap.utils.consts import SF_COLS
from gaitmap.utils.datatype_helper import (
    is_multi_sensor_orientation_list,
    is_multi_sensor_position_list,
    is_single_sensor_orientation_list,
    is_single_sensor_position_list,
)
from gaitmap.utils.exceptions import ValidationError


class TestIODataStructures:
    wrapper_class: Type[BaseTrajectoryReconstructionWrapper]
    example_region: Dict[str, pd.DataFrame]
    key: Literal["s_id", "roi_id"]
    output_list_type: Literal["roi", "stride"]

    @pytest.fixture(params=(StrideLevelTrajectory, RegionLevelTrajectory), autouse=True)
    def select_wrapper(self, healthy_example_stride_events, request):
        self.wrapper_class = request.param
        if self.wrapper_class == RegionLevelTrajectory:
            self.example_region = {
                k: v.rename(columns={"s_id": "roi_id"}) for k, v in healthy_example_stride_events.items()
            }
            self.output_list_type = "roi"
            self.key = "roi_id"
        else:
            self.example_region = healthy_example_stride_events
            self.output_list_type = "stride"
            self.key = "s_id"

    def test_invalid_input_data(self, healthy_example_imu_data):
        """Test if error is raised correctly on invalid input data type"""
        data = healthy_example_imu_data
        stride_list = self.example_region
        fake_data = pd.DataFrame({"a": [0, 1, 2], "b": [3, 4, 5]})
        fake_stride_list = {
            "a": pd.DataFrame(data=[[0, 1, 2]], columns=["stride", "begin", "stop"]),
            "b": pd.DataFrame(data=[[0, 1, 2]], columns=["stride", "begin", "stop"]),
        }

        gyr_int = self.wrapper_class(align_window_width=8)
        with pytest.raises(ValidationError, match=r".*neither single- or multi-sensor data"):
            gyr_int.estimate(fake_data, stride_list, 204.8)
        with pytest.raises(ValidationError, match=r".*neither a single- or a multi-sensor "):
            gyr_int.estimate(data, fake_stride_list, 204.8)

    @pytest.mark.parametrize("method", ("ori_method", "pos_method"))
    def test_invalid_input_method(self, healthy_example_imu_data, method):
        """Test if correct errors are raised for invalid pos and ori methods."""
        instance = self.wrapper_class(**{method: "wrong"})
        with pytest.raises(ValueError) as e:
            instance.estimate(healthy_example_imu_data, self.example_region, 204.8)
        assert method in str(e)

    @pytest.mark.parametrize("method", ("ori_method", "pos_method"))
    def test_only_pos_or_ori_provided(self, healthy_example_imu_data, method):
        instance = self.wrapper_class(**{method: None})
        with pytest.raises(ValueError) as e:
            instance.estimate(healthy_example_imu_data, self.example_region, 204.8)
        assert "either a `ori` and a `pos` method" in str(e)

    @pytest.mark.parametrize("method", ("ori_method", "pos_method"))
    def test_trajectory_warning(self, healthy_example_imu_data, method):
        instance = self.wrapper_class(**{method: RtsKalman()})
        with pytest.warns(UserWarning) as w:
            instance.estimate(
                healthy_example_imu_data["left_sensor"], self.example_region["left_sensor"].loc[[0]], 204.8
            )
        assert "trajectory method as ori or pos method" in str(w[0])

    def test_passed_both_warning(self, healthy_example_imu_data):
        """Test that a warning is raised when passing ori and pos and trajectory emthods all at one.

        This will happen by default, when leaving ori and pos as default values
        """
        instance = self.wrapper_class(trajectory_method=RtsKalman())
        with pytest.warns(UserWarning) as w:
            instance.estimate(
                healthy_example_imu_data["left_sensor"], self.example_region["left_sensor"].loc[[0]], 204.8
            )
        assert "You provided a trajectory method AND an ori or pos method." in str(w[0])

    def test_single_sensor_output(self, healthy_example_imu_data, snapshot):
        test_stride_events = self.example_region["left_sensor"].iloc[:3]
        test_data = healthy_example_imu_data["left_sensor"].iloc[: int(test_stride_events.iloc[-1]["end"])]

        instance = self.wrapper_class()
        instance.estimate(test_data, test_stride_events, 204.8)

        assert is_single_sensor_orientation_list(instance.orientation_, self.output_list_type)
        assert is_single_sensor_position_list(instance.position_, self.output_list_type)
        assert_array_equal(
            instance.orientation_.reset_index()[self.key].unique(), test_stride_events[self.key].unique()
        )
        assert_array_equal(instance.position_.reset_index()[self.key].unique(), test_stride_events[self.key].unique())
        assert_array_equal(instance.velocity_.reset_index()[self.key].unique(), test_stride_events[self.key].unique())

        for _, s in test_stride_events.iterrows():
            assert len(instance.orientation_.loc[s[self.key]]) == s["end"] - s["start"] + 1
            assert len(instance.position_.loc[s[self.key]]) == s["end"] - s["start"] + 1
            assert len(instance.velocity_.loc[s[self.key]]) == s["end"] - s["start"] + 1

        first_last_stride = test_stride_events.iloc[[0, -1]][self.key]
        snapshot.assert_match(instance.orientation_.loc[first_last_stride], "ori")
        snapshot.assert_match(instance.position_.loc[first_last_stride], "pos")

    def test_single_sensor_output_empty_stride_list(self, healthy_example_imu_data):
        empty_stride_events = pd.DataFrame(columns=self.example_region["left_sensor"].columns)
        test_data = healthy_example_imu_data["left_sensor"]

        instance = self.wrapper_class()
        instance.estimate(test_data, empty_stride_events, 204.8)

        assert is_single_sensor_orientation_list(instance.orientation_, self.output_list_type)
        assert is_single_sensor_position_list(instance.position_, self.output_list_type)
        assert len(instance.orientation_) == 0
        assert len(instance.position_) == 0

    def test_multi_sensor_output(self, healthy_example_imu_data, snapshot):
        test_stride_events = self.example_region
        test_data = healthy_example_imu_data

        instance = self.wrapper_class()
        instance.estimate(test_data, test_stride_events, 204.8)

        assert is_multi_sensor_orientation_list(instance.orientation_, self.output_list_type)
        assert is_multi_sensor_position_list(instance.position_, self.output_list_type)
        for sensor in test_stride_events.keys():
            assert_array_equal(
                instance.orientation_[sensor].reset_index()[self.key].unique(),
                test_stride_events[sensor][self.key].unique(),
            )
            assert_array_equal(
                instance.position_[sensor].reset_index()[self.key].unique(),
                test_stride_events[sensor][self.key].unique(),
            )
            assert_array_equal(
                instance.velocity_[sensor].reset_index()[self.key].unique(),
                test_stride_events[sensor][self.key].unique(),
            )

            for _, s in test_stride_events[sensor].iterrows():
                assert len(instance.orientation_[sensor].loc[s[self.key]]) == s["end"] - s["start"] + 1
                assert len(instance.position_[sensor].loc[s[self.key]]) == s["end"] - s["start"] + 1
                assert len(instance.velocity_[sensor].loc[s[self.key]]) == s["end"] - s["start"] + 1

            first_last_stride = test_stride_events[sensor].iloc[[0, -1]][self.key]
            snapshot.assert_match(instance.orientation_[sensor].loc[first_last_stride], "ori_{}".format(sensor))
            snapshot.assert_match(instance.position_[sensor].loc[first_last_stride], "pos_{}".format(sensor))


class TestInitCalculation:
    """Test the calculation of initial rotations per stride.

    No complicated tests here, as this uses `get_gravity_rotation`, which is well tested
    """

    def test_calc_initial_dummy(self):
        """No rotation expected as already aligned."""
        dummy_data = pd.DataFrame(np.repeat(np.array([0, 0, 1, 0, 0, 0])[None, :], 20, axis=0), columns=SF_COLS)
        start_ori = _initial_orientation_from_start(dummy_data, 10, 8)
        assert_array_equal(start_ori.as_quat(), Rotation.identity().as_quat())

    @pytest.mark.parametrize("start", [0, 99])
    def test_start_of_stride_equals_start_or_end_of_data(self, start):
        """If start is to close to the start or the end of the data a warning is emitted."""
        dummy_data = pd.DataFrame(np.repeat(np.array([0, 0, 1, 0, 0, 0])[None, :], 100, axis=0), columns=SF_COLS)
        with pytest.warns(UserWarning) as w:
            _initial_orientation_from_start(dummy_data, start, 8)

        assert "complete window length" in str(w[0])

    def test_only_single_value(self):
        dummy_data = pd.DataFrame(np.repeat(np.array([0, 0, 1, 0, 0, 0])[None, :], 20, axis=0), columns=SF_COLS)
        start_ori = _initial_orientation_from_start(dummy_data, 10, 0)
        assert_array_equal(start_ori.as_quat(), Rotation.identity().as_quat())
