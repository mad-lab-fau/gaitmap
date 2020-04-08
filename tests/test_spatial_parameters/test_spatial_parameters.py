from gaitmap.parameters.spatial_parameters import SpatialParameterCalculation

import pytest
from pandas.testing import assert_frame_equal
import pandas as pd


@pytest.fixture
def single_sensor_stride_list():
    stride_events_list = pd.DataFrame(columns=["s_id", "ic", "tc", "pre_ic", "gsd_id", "min_vel", "start", "end"])
    stride_events_list["s_id"] = [0, 1, 2]
    stride_events_list["ic"] = [500.0, 700.0, 1000.0]
    stride_events_list["tc"] = [400.0, 600.0, 800.0]
    stride_events_list["pre_ic"] = [300.0, 500.0, 700.0]
    stride_events_list["start"] = [350, 550, 750]
    stride_events_list["min_vel"] = stride_events_list["start"]
    return stride_events_list


@pytest.fixture
def single_sensor_position_list():
    position_list = pd.DataFrame(columns=["s_id", "position", "velocity"])
    position_list["s_id"] = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    position_list["position"] = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
                                 [1, 0, 0], [1, 0, 0], [1, 0, 0]]
    position_list["velocity"] = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
                                 [1, 0, 0], [1, 0, 0], [1, 0, 0]]
    return position_list


@pytest.fixture
def single_sensor_orientation_list():
    orientation_list = pd.DataFrame(columns=["s_id", "orientation"])
    orientation_list["s_id"] = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    orientation_list["orientation"] = [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
                                       [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
    return orientation_list


@pytest.fixture
def spatial_parameters_single_sensor():
    spatial_parameters = pd.DataFrame(columns=["s_id", "stride_length", "gait_velocity"])
    spatial_parameters["s_id"] = [0, 1, 2]
    spatial_parameters["stride_length"] = [1.0, 1.0, 1.0]
    spatial_parameters["gait_velocity"] = [0.5, 0.5, 0.3333333333333333]
    return spatial_parameters


class TestSpatialParameterCalculation:
    """Test temporal parameters calculation."""

    def test_single_sensor(self, single_sensor_stride_list, single_sensor_position_list,
                           single_sensor_orientation_list, spatial_parameters_single_sensor):
        """Test calculate spatial parameters for single sensor """
        t = SpatialParameterCalculation()
        t.calculate(single_sensor_stride_list, single_sensor_position_list, single_sensor_orientation_list, 100)
        assert_frame_equal(t.parameters_, spatial_parameters_single_sensor)

    def test_multiple_sensor(self, single_sensor_stride_list, single_sensor_position_list
                             , single_sensor_orientation_list,
                             spatial_parameters_single_sensor):
        """Test calculate temporal parameters for single sensor and single stride"""
        stride_events_list = {"sensor1": single_sensor_stride_list, "sensor2": single_sensor_stride_list}
        position_list = {"sensor1": single_sensor_position_list, "sensor2": single_sensor_position_list}
        orientation_list = {"sensor1": single_sensor_orientation_list, "sensor2": single_sensor_orientation_list}
        t = SpatialParameterCalculation()
        t.calculate(stride_events_list, position_list, orientation_list, 100)
        spatial_parameters_multiple_sensor = {"sensor1": spatial_parameters_single_sensor,
                                              "sensor2": spatial_parameters_single_sensor}
        for sensor in t.parameters_:
            assert_frame_equal(t.parameters_[sensor], spatial_parameters_multiple_sensor[sensor])

