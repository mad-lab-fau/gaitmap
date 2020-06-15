from gaitmap.base import BaseType
from gaitmap.parameters.temporal_parameters import TemporalParameterCalculation

import pytest
from pandas.testing import assert_frame_equal
import pandas as pd

from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin


@pytest.fixture
def single_stride_list():
    stride_events_list = pd.DataFrame(columns=["s_id", "ic", "tc", "pre_ic", "gsd_id", "min_vel", "start", "end"])
    stride_events_list["s_id"] = [0]
    stride_events_list["ic"] = [500.0]
    stride_events_list["tc"] = [400.0]
    stride_events_list["pre_ic"] = [300.0]
    stride_events_list["start"] = [350]
    stride_events_list["min_vel"] = stride_events_list["start"]
    return stride_events_list


@pytest.fixture
def multiple_stride_list():
    stride_events_list = pd.DataFrame(columns=["s_id", "ic", "tc", "pre_ic", "gsd_id", "min_vel", "start", "end"])
    stride_events_list["s_id"] = [0, 1, 2]
    stride_events_list["ic"] = [500.0, 700.0, 1000.0]
    stride_events_list["tc"] = [400.0, 600.0, 800.0]
    stride_events_list["pre_ic"] = [300.0, 500.0, 700.0]
    stride_events_list["start"] = [350, 550, 750]
    stride_events_list["min_vel"] = stride_events_list["start"]
    return stride_events_list


@pytest.fixture
def temporal_parameters_multiple_strides():
    temporal_parameters = pd.DataFrame(columns=["s_id", "stride_time", "swing_time", "stance_time"])
    temporal_parameters["s_id"] = [0, 1, 2]
    temporal_parameters["stride_time"] = [2.0, 2.0, 3.0]
    temporal_parameters["swing_time"] = [1.0, 1.0, 2.0]
    temporal_parameters["stance_time"] = [1.0, 1.0, 1.0]
    return temporal_parameters.set_index("s_id")


@pytest.fixture
def temporal_parameters_single_stride():
    temporal_parameters = pd.DataFrame(columns=["s_id", "stride_time", "swing_time", "stance_time"])
    temporal_parameters["s_id"] = [0]
    temporal_parameters["stride_time"] = [2.0]
    temporal_parameters["swing_time"] = [1.0]
    temporal_parameters["stance_time"] = [1.0]
    return temporal_parameters.set_index("s_id")


class TestMetaFunctionality(TestAlgorithmMixin):
    algorithm_class = TemporalParameterCalculation
    __test__ = True

    @pytest.fixture()
    def after_action_instance(self, multiple_stride_list, temporal_parameters_multiple_strides) -> BaseType:
        stride_events_list = multiple_stride_list
        t = TemporalParameterCalculation()
        t.calculate(stride_events_list, 100)
        return t


class TestTemporalParameterCalculation:
    """Test temporal parameters calculation."""

    def test_single_sensor_multiple_strides(self, multiple_stride_list, temporal_parameters_multiple_strides):
        """Test calculate temporal parameters for single sensor """
        stride_events_list = multiple_stride_list
        expected_temporal_parameters = temporal_parameters_multiple_strides
        t = TemporalParameterCalculation()
        t.calculate(stride_events_list, 100)
        assert_frame_equal(t.parameters_, expected_temporal_parameters)

    def test_single_sensor_single_stride(self, single_stride_list, temporal_parameters_single_stride):
        """Test calculate temporal parameters for single sensor and single stride"""
        stride_events_list = single_stride_list
        expected_temporal_parameters = temporal_parameters_single_stride
        t = TemporalParameterCalculation()
        t.calculate(stride_events_list, 100)
        assert_frame_equal(t.parameters_, expected_temporal_parameters)

    def test_multiple_sensor(self, multiple_stride_list, temporal_parameters_multiple_strides):
        """Test calculate temporal parameters for multiple sensors , multiple strides for all sensors"""
        stride_events_list1 = multiple_stride_list
        stride_events_list = {"sensor1": stride_events_list1, "sensor2": stride_events_list1}
        expected_temporal_parameters = temporal_parameters_multiple_strides
        expected_temporal_parameters = {
            "sensor1": expected_temporal_parameters,
            "sensor2": expected_temporal_parameters,
        }
        t = TemporalParameterCalculation()
        t.calculate(stride_events_list, 100)
        for sensor in t.parameters_:
            assert_frame_equal(t.parameters_[sensor], expected_temporal_parameters[sensor])

    def test_multiple_sensor_single_stride(self, single_stride_list, temporal_parameters_single_stride):
        """Test calculate temporal parameters for multiple sensors , single stride for all sensors"""
        stride_events_list1 = single_stride_list
        stride_events_list = {"sensor1": stride_events_list1, "sensor2": stride_events_list1}
        expected_temporal_parameters = temporal_parameters_single_stride
        expected_temporal_parameters = {
            "sensor1": expected_temporal_parameters,
            "sensor2": expected_temporal_parameters,
        }
        t = TemporalParameterCalculation()
        t.calculate(stride_events_list, 100)
        for sensor in t.parameters_:
            assert_frame_equal(t.parameters_[sensor], expected_temporal_parameters[sensor])

    def test_multiple_sensor_single_stride_for_one_sensor(
        self,
        multiple_stride_list,
        single_stride_list,
        temporal_parameters_single_stride,
        temporal_parameters_multiple_strides,
    ):
        """Test calculate temporal parameters for multiple sensors , single stride for one sensor"""
        stride_events_list1 = multiple_stride_list
        stride_events_list2 = single_stride_list
        stride_events_list = {"sensor1": stride_events_list1, "sensor2": stride_events_list2}
        expected_temporal_parameters_sensor1 = temporal_parameters_multiple_strides
        expected_temporal_parameters_sensor2 = temporal_parameters_single_stride
        expected_temporal_parameters = {
            "sensor1": expected_temporal_parameters_sensor1,
            "sensor2": expected_temporal_parameters_sensor2,
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
