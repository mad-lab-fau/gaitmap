import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_series_equal

from gaitmap.base import BaseType
from gaitmap.parameters.spatial_parameters import SpatialParameterCalculation, _calc_stride_length, _calc_gait_velocity
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
def single_sensor_stride_length():
    out = pd.Series([2, np.sqrt(8), 0], index=[0, 1, 2])
    out.index.name = "s_id"
    return out


@pytest.fixture()
def single_sensor_gait_speed(single_sensor_stride_length, single_sensor_stride_time):
    return single_sensor_stride_length / single_sensor_stride_time


@pytest.fixture
def single_sensor_orientation_list():
    orientation_list = pd.DataFrame(columns=["s_id", "sample", "qx", "qy", "qz", "qw"])
    orientation_list["s_id"] = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    orientation_list["sample"] = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    orientation_list["qx"] = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    orientation_list["qy"] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    orientation_list["qz"] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    orientation_list["qw"] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    return orientation_list


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
    def test_stride_length(self, single_sensor_position_list, single_sensor_stride_length):
        assert_series_equal(
            _calc_stride_length(single_sensor_position_list.set_index(["s_id", "sample"])), single_sensor_stride_length
        )

    def test_gait_speed(self, single_sensor_stride_length, single_sensor_stride_time, single_sensor_gait_speed):
        assert_series_equal(
            _calc_gait_velocity(single_sensor_stride_length, single_sensor_stride_time), single_sensor_gait_speed
        )


class TestSpatialParameterCalculation:
    """Test temporal parameters calculation."""

    def test_single_sensor(
        self, single_sensor_stride_list, single_sensor_position_list, single_sensor_orientation_list
    ):
        """Test calculate spatial parameters for single sensor """
        t = SpatialParameterCalculation()
        t.calculate(single_sensor_stride_list, single_sensor_position_list, single_sensor_orientation_list, 100)
        # TODO: Make into actual test
        return None

    def test_multiple_sensor(
        self, single_sensor_stride_list, single_sensor_position_list, single_sensor_orientation_list
    ):
        """Test calculate temporal parameters for single sensor and single stride"""
        stride_events_list = {"sensor1": single_sensor_stride_list, "sensor2": single_sensor_stride_list}
        position_list = {"sensor1": single_sensor_position_list, "sensor2": single_sensor_position_list}
        orientation_list = {"sensor1": single_sensor_orientation_list, "sensor2": single_sensor_orientation_list}
        t = SpatialParameterCalculation()
        t.calculate(stride_events_list, position_list, orientation_list, 100)
        # TODO: make into actual test
