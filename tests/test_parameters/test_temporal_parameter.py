import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gaitmap.base import BaseType
from gaitmap.parameters import TemporalParameterCalculation
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin


def min_vel_stride_list():
    stride_events_list = pd.DataFrame(columns=["s_id", "ic", "tc", "pre_ic", "min_vel", "start", "end"])
    stride_events_list["s_id"] = [0, 1, 2]
    stride_events_list["ic"] = [500.0, 700.0, 1000.0]
    stride_events_list["tc"] = [400.0, 600.0, 800.0]
    stride_events_list["pre_ic"] = [300.0, 500.0, 700.0]
    stride_events_list["start"] = [350, 550, 750]
    stride_events_list["end"] = [450, 650, 850]
    stride_events_list["min_vel"] = stride_events_list["start"]
    stride_events_list = stride_events_list.set_index("s_id")

    temporal_parameters = pd.DataFrame(columns=["s_id", "stride_time", "swing_time", "stance_time"])
    temporal_parameters["s_id"] = [0, 1, 2]
    temporal_parameters["stride_time"] = [2.0, 2.0, 3.0]
    temporal_parameters["swing_time"] = [1.0, 1.0, 2.0]
    temporal_parameters["stance_time"] = [1.0, 1.0, 1.0]
    temporal_parameters = temporal_parameters.set_index("s_id")
    return stride_events_list, temporal_parameters


def ic_stride_list():
    stride_events_list = pd.DataFrame(columns=["s_id", "ic", "tc", "min_vel", "start", "end"])
    stride_events_list["s_id"] = [0, 1, 2]
    stride_events_list["start"] = [350, 550, 750]
    stride_events_list["ic"] = stride_events_list["start"]
    stride_events_list["min_vel"] = [400.0, 600.0, 800.0]
    stride_events_list["tc"] = [450, 650, 850]
    stride_events_list["end"] = [500.0, 700.0, 1000.0]
    stride_events_list = stride_events_list.set_index("s_id")

    temporal_parameters = pd.DataFrame(columns=["s_id", "stride_time", "swing_time", "stance_time"])
    temporal_parameters["s_id"] = [0, 1, 2]
    temporal_parameters["stride_time"] = [1.5, 1.5, 2.5]
    temporal_parameters["swing_time"] = [0.5, 0.5, 1.5]
    temporal_parameters["stance_time"] = [1.0, 1.0, 1.0]
    temporal_parameters = temporal_parameters.set_index("s_id")
    return stride_events_list, temporal_parameters


class TestMetaFunctionality(TestAlgorithmMixin):
    algorithm_class = TemporalParameterCalculation
    __test__ = True

    @pytest.fixture()
    def after_action_instance(self, min_vel_stride_list) -> BaseType:
        stride_events_list, _ = min_vel_stride_list
        t = TemporalParameterCalculation()
        t.calculate(stride_events_list, 100)
        return t


@pytest.mark.parametrize("stride_list_type", ["min_vel", "ic"])
class TestTemporalParameterCalculation:
    """Test temporal parameters calculation."""

    @pytest.fixture()
    def stride_list(self, stride_list_type):
        if stride_list_type == "min_vel":
            return min_vel_stride_list()
        elif stride_list_type == "ic":
            return ic_stride_list()

    def test_single_sensor_multiple_strides(self, stride_list, stride_list_type):
        """Test calculate temporal parameters for single sensor."""
        stride_events_list, temporal_parameters = stride_list
        t = TemporalParameterCalculation(expected_stride_type=stride_list_type)
        t.calculate(stride_events_list, 100)
        assert_frame_equal(t.parameters_, temporal_parameters)

    def test_multiple_sensor(self, stride_list, stride_list_type):
        """Test calculate temporal parameters for multiple sensors , multiple strides for all sensors."""
        stride_events_list1, temporal_parameters = stride_list
        stride_events_list = {"sensor1": stride_events_list1.iloc[:2], "sensor2": stride_events_list1}
        expected_temporal_parameters = {
            "sensor1": temporal_parameters.iloc[:2],
            "sensor2": temporal_parameters,
        }
        t = TemporalParameterCalculation(expected_stride_type=stride_list_type)
        t.calculate(stride_events_list, 100)
        for sensor in t.parameters_:
            assert_frame_equal(t.parameters_[sensor], expected_temporal_parameters[sensor])


class TestTemporalParametersIcStrideList:
    def test_single_sensor_multiple_strides(self, min_vel_stride_list):
        """Test calculate temporal parameters for single sensor."""
        stride_events_list, temporal_parameters = min_vel_stride_list
        t = TemporalParameterCalculation()
        t.calculate(stride_events_list, 100)
        assert_frame_equal(t.parameters_, temporal_parameters)

    def test_multiple_sensor(self, min_vel_stride_list):
        """Test calculate temporal parameters for multiple sensors , multiple strides for all sensors."""
        stride_events_list1, temporal_parameters = min_vel_stride_list
        stride_events_list = {"sensor1": stride_events_list1.iloc[:2], "sensor2": stride_events_list1}
        expected_temporal_parameters = {
            "sensor1": temporal_parameters.iloc[:2],
            "sensor2": temporal_parameters,
        }
        t = TemporalParameterCalculation()
        t.calculate(stride_events_list, 100)
        for sensor in t.parameters_:
            assert_frame_equal(t.parameters_[sensor], expected_temporal_parameters[sensor])


class TestTemporalParameterRegression:
    def test_regression_on_example_data(self, healthy_example_stride_events, snapshot):
        healthy_example_stride_events = healthy_example_stride_events["left_sensor"]
        t = TemporalParameterCalculation()
        t.calculate(healthy_example_stride_events, 204.8)
        snapshot.assert_match(t.parameters_)
