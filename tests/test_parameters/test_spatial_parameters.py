import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_series_equal

from gaitmap.base import BaseType
from gaitmap.parameters import SpatialParameterCalculation
from gaitmap.parameters._spatial_parameters import (
    _calc_arc_length,
    _calc_gait_velocity,
    _calc_stride_length,
    _calc_turning_angle,
    _compute_sole_angle_course,
)
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin


@pytest.fixture
def single_sensor_stride_list():
    stride_events_list = pd.DataFrame(columns=["s_id", "ic", "tc", "pre_ic", "gsd_id", "min_vel", "start", "end"])
    stride_events_list["s_id"] = [0, 1, 2]
    stride_events_list["ic"] = [3.0, 5.0, 7.0]
    stride_events_list["tc"] = [2.0, 4.0, 6.0]
    stride_events_list["pre_ic"] = [1.0, 3.0, 5.0]
    stride_events_list["start"] = [2, 4, 5]
    stride_events_list["min_vel"] = stride_events_list["start"]
    return stride_events_list


@pytest.fixture()
def single_sensor_stride_time():
    out = pd.Series([2, 2, 2], index=[0, 1, 2])
    out.index.name = "s_id"
    return out


@pytest.fixture
def single_sensor_position_list():
    position_list = pd.DataFrame(columns=["s_id", "sample", "pos_x", "pos_y", "pos_z"])
    position_list["s_id"] = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    position_list["sample"] = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    position_list["pos_x"] = [1, 2, 3, 1, 2, 3, 1, 1, 1]
    position_list["pos_y"] = [0, 0, 0, 0, 1, 2, 0, 0, 0]
    position_list["pos_z"] = [0, 0, 0, 0, 0, 0, 1, 2, 3]
    return position_list


@pytest.fixture()
def single_sensor_position_list_with_index(single_sensor_position_list):
    return single_sensor_position_list.set_index(["s_id", "sample"])


@pytest.fixture()
def single_sensor_stride_length():
    out = pd.Series([2, np.sqrt(8), 0], index=[0, 1, 2])
    out.index.name = "s_id"
    return out


@pytest.fixture()
def single_sensor_arc_length():
    out = pd.Series([2, 2 * np.sqrt(2), 2], index=[0, 1, 2])
    out.index.name = "s_id"
    return out


@pytest.fixture()
def single_sensor_gait_speed(single_sensor_stride_length, single_sensor_stride_time):
    return single_sensor_stride_length / single_sensor_stride_time


@pytest.fixture
def single_sensor_orientation_list():
    orientation_list = pd.DataFrame(columns=["s_id", "sample", "q_x", "q_y", "q_z", "q_w"])
    orientation_list["s_id"] = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    orientation_list["sample"] = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    orientation_list["q_x"] = [0, 0, 0, 0, np.sqrt(2), 0, 0, 0, 0]
    orientation_list["q_y"] = [0, np.sqrt(2), 0, 0, 0, 0, 0, 0, 0]
    orientation_list["q_z"] = [0, 0, 0, 0, 0, np.sqrt(2), 0, np.sqrt(2), np.sqrt(2)]
    orientation_list["q_w"] = [1, np.sqrt(2), 1, 1, np.sqrt(2), np.sqrt(2), 1, np.sqrt(2), -np.sqrt(2)]
    return orientation_list


@pytest.fixture()
def single_sensor_orientation_list_with_index(single_sensor_orientation_list):
    return single_sensor_orientation_list.set_index(["s_id", "sample"])


@pytest.fixture()
def single_sensor_turning_angle():
    out = pd.Series([0.0, 90, -90], index=[0, 1, 2])
    out.index.name = "s_id"
    return out


@pytest.fixture()
def single_sensor_sole_angle_course():
    index = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    sample = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    index = pd.MultiIndex.from_arrays([index, sample], names=["s_id", "sample"])
    angle = [0, -90.0, 0, 0, 0, 0, 0, 0, 0]
    return pd.Series(angle, index=index)


class TestMetaFunctionality(TestAlgorithmMixin):
    algorithm_class = SpatialParameterCalculation
    __test__ = True

    @pytest.fixture()
    def after_action_instance(
        self, single_sensor_stride_list, single_sensor_position_list, single_sensor_orientation_list
    ) -> BaseType:
        t = SpatialParameterCalculation()
        t.calculate(single_sensor_stride_list, single_sensor_position_list, single_sensor_orientation_list, 100)
        return t


class TestIndividualParameter:
    def test_stride_length(self, single_sensor_position_list_with_index, single_sensor_stride_length):
        assert_series_equal(_calc_stride_length(single_sensor_position_list_with_index), single_sensor_stride_length)

    def test_gait_speed(self, single_sensor_stride_length, single_sensor_stride_time, single_sensor_gait_speed):
        assert_series_equal(
            _calc_gait_velocity(single_sensor_stride_length, single_sensor_stride_time), single_sensor_gait_speed
        )

    def test_arc_length(self, single_sensor_position_list_with_index, single_sensor_arc_length):
        assert_series_equal(_calc_arc_length(single_sensor_position_list_with_index), single_sensor_arc_length)

    def test_turning_angle(self, single_sensor_orientation_list_with_index, single_sensor_turning_angle):
        assert_series_equal(
            _calc_turning_angle(single_sensor_orientation_list_with_index),
            single_sensor_turning_angle,
            check_exact=False,
        )

    def test_turning_angle_empty_orientation(self):
        """Test the turning angle computation in case of empty orientation input.

        For scipy<=1.5.4 this produced an empty Series automatically. For scipy>1.6.0 we need to handle the case of
        an empty input manually. The expected output is defined based on the behavior for scipy==1.5.4.
        """
        empty_orientations = pd.DataFrame(columns=["q_x", "q_y", "q_z", "q_w"])
        empty_orientations.index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=["s_id", "sample"])

        expected_out = pd.Series()
        expected_out.index.name = "s_id"

        assert_series_equal(
            _calc_turning_angle(empty_orientations),
            expected_out,
            check_exact=False,
        )

    def test_sole_angle(self, single_sensor_orientation_list_with_index, single_sensor_sole_angle_course):
        assert_series_equal(
            _compute_sole_angle_course(single_sensor_orientation_list_with_index), single_sensor_sole_angle_course
        )

    def test_sole_angle_empty_orientation(self):
        """Test the sole angle computation in case of empty orientation input.

        For scipy<=1.5.4 this produced an empty Series automatically. For scipy>1.6.0 we need to handle the case of
        an empty input manually. The expected output is defined based on the behavior for scipy==1.5.4.
        """
        empty_orientations = pd.DataFrame(columns=["q_x", "q_y", "q_z", "q_w"])
        empty_orientations.index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=["s_id", "sample"])

        expected_out = pd.Series()
        expected_out.index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=["s_id", "sample"])

        assert_series_equal(
            _compute_sole_angle_course(empty_orientations),
            expected_out,
            check_exact=False,
        )


class TestSpatialParameterCalculation:
    """Test spatial parameters calculation."""

    parameters = [
        "stride_length",
        "gait_velocity",
        "ic_angle",
        "tc_angle",
        "turning_angle",
        "arc_length",
        "max_sensor_lift",
        "max_lateral_excursion",
    ]

    def test_single_sensor(
        self, single_sensor_stride_list, single_sensor_position_list, single_sensor_orientation_list
    ):
        """Test calculate spatial parameters for single sensor"""
        t = SpatialParameterCalculation()
        t.calculate(single_sensor_stride_list, single_sensor_position_list, single_sensor_orientation_list, 100)
        # Test that all parameters are at least theoretically calculated
        assert set(t.parameters_.columns) == set(self.parameters)
        assert len(t.parameters_) == len(single_sensor_stride_list)
        assert len(t.sole_angle_course_) == len(single_sensor_orientation_list)

    def test_multiple_sensor(
        self, single_sensor_stride_list, single_sensor_position_list, single_sensor_orientation_list
    ):
        """Test calculate spatial parameters for single sensor and single stride"""
        stride_events_list = {"sensor1": single_sensor_stride_list, "sensor2": single_sensor_stride_list}
        position_list = {"sensor1": single_sensor_position_list, "sensor2": single_sensor_position_list}
        orientation_list = {"sensor1": single_sensor_orientation_list, "sensor2": single_sensor_orientation_list}
        t = SpatialParameterCalculation()
        t.calculate(stride_events_list, position_list, orientation_list, 100)
        assert isinstance(t.parameters_, dict)
        assert set(t.parameters_.keys()) == {"sensor1", "sensor2"}
        for sensor in t.parameters_.values():
            assert set(sensor.columns) == set(self.parameters)
            assert len(sensor) == len(single_sensor_stride_list)
        assert isinstance(t.sole_angle_course_, dict)
        assert set(t.sole_angle_course_.keys()) == {"sensor1", "sensor2"}
        for sensor in t.sole_angle_course_.values():
            assert len(sensor) == len(single_sensor_orientation_list)

    @pytest.mark.parametrize(
        "exclude, expected_missing",
        [
            (("ic", "pre_ic"), ["ic_angle", "gait_velocity"]),
            (("tc",), ["tc_angle"]),
            (("ic", "pre_ic", "tc"), ["ic_angle", "tc_angle", "gait_velocity"]),
        ],
    )
    def test_partial_info(
        self,
        exclude,
        expected_missing,
        single_sensor_stride_list,
        single_sensor_position_list,
        single_sensor_orientation_list,
    ):
        """Test that it is possible to calculate spatial parameters with partial information."""
        stride_list = single_sensor_stride_list.drop(list(exclude), axis=1)
        stride_events_list = {"sensor1": stride_list, "sensor2": stride_list}
        position_list = {"sensor1": single_sensor_position_list, "sensor2": single_sensor_position_list}
        orientation_list = {"sensor1": single_sensor_orientation_list, "sensor2": single_sensor_orientation_list}
        t = SpatialParameterCalculation()
        t.calculate(stride_events_list, position_list, orientation_list, 100)
        assert isinstance(t.parameters_, dict)
        assert set(t.parameters_.keys()) == {"sensor1", "sensor2"}
        for sensor in t.parameters_.values():
            assert set(sensor.columns) == set(self.parameters) - set(expected_missing)
            assert len(sensor) == len(single_sensor_stride_list)
        assert isinstance(t.sole_angle_course_, dict)
        assert set(t.sole_angle_course_.keys()) == {"sensor1", "sensor2"}
        for sensor in t.sole_angle_course_.values():
            assert len(sensor) == len(single_sensor_orientation_list)


class TestSpatialParameterRegression:
    def test_regression_on_example_data(
        self, healthy_example_orientation, healthy_example_position, healthy_example_stride_events, snapshot
    ):
        # Convert stride list back to mocap samples:
        healthy_example_stride_events["left_sensor"][["start", "end", "tc", "ic", "min_vel", "pre_ic"]] *= 100 / 204.8
        healthy_example_stride_events["right_sensor"][["start", "end", "tc", "ic", "min_vel", "pre_ic"]] *= 100 / 204.8
        t = SpatialParameterCalculation()
        t.calculate(healthy_example_stride_events, healthy_example_position, healthy_example_orientation, 100)
        snapshot.assert_match(t.parameters_["left_sensor"])
