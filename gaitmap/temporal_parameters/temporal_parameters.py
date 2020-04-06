"""Calculate temporal parameters algorithm."""
import numpy as np
import pandas as pd

from gaitmap.base import BaseType, BaseTemporalParameterCalculation
from gaitmap.event_detection.rampp_event_detection import RamppEventDetection


class TemporalParameterCalculation(BaseTemporalParameterCalculation):
    """This class is responsible for calculating temporal parameters of strides.

    Parameters
    ----------
    stride_event_list
            Gait events for each stride obtained from Rampp event detection
    sampling_rate_hz
        The sampling rate of the data signal.
    parameters_
        Data frame containing temporal parameters for each stride in case of single sensor or dictionary of data frames in multi sensor

    Attributes
    ----------
    stride_id_
        Array of stride ids that will be needed further for combining other parameters
    time_stamp_
        Array of time stamps of each stride that could be useful for home monitoring
    stride_time_
        Array of stride time for each stride
    swing_time_
       Array of swing time for each stride
    stance_time_
        Array of stance time for each stride

    """

    parameters_: dict  # Results are stored here

    def _calculate_single_sensor(self: BaseType, stride_event_list: pd.DataFrame, sampling_rate_hz: float) -> BaseType:
        """Find temporal parameters in strides after segmentation and detecting events of each stride in case of dingle sensor.

        Parameters
        ----------
        stride_event_list
            Gait events for each stride obtained from event detection
        sampling_rate_hz
            The sampling rate of the data signal.

        Returns
        -------
            self
                The class instance with temporal parameters populated in parameters_

        """
        stride_id_ = np.ndarray
        time_stamp_ = np.ndarray
        stride_time_ = np.ndarray
        swing_time_ = np.ndarray
        stance_time_ = np.ndarray
        stride_id_ = stride_event_list["s_id"]
        time_stamp_ = _calc_time_stamp_(stride_event_list["ic"], sampling_rate_hz)
        stride_time_ = _calc_stride_time(stride_event_list["ic"], stride_event_list["pre_ic"], sampling_rate_hz)
        swing_time_ = _calc_swing_time(stride_event_list["ic"], stride_event_list["tc"], sampling_rate_hz)
        stance_time_ = [stride_time - swing_time for stride_time, swing_time in zip(stride_time_, swing_time_)]
        stride_parameter_dict = {
            "stride_id": stride_id_,
            "time_stamp": time_stamp_,
            "stride_time": stride_time_,
            "swing_time": swing_time_,
            "stance_time": stance_time_,
        }
        self.parameters_ = pd.DataFrame(stride_parameter_dict)
        return self

    def _calculate_multiple_sensor(self: BaseType, stride_event_list: dict, sampling_rate_hz: float) -> BaseType:
        """Find temporal parameters of each stride in case of multiple sensors.

        Parameters
        ----------
        stride_event_list
            Gait events for each stride obtained from event detection
        sampling_rate_hz
            The sampling rate of the data signal.

        Returns
        -------
            self
                The class instance with temporal parameters populated in parameters_

        """
        self.parameters_ = {}
        for sensor in stride_event_list:
            self.parameters_[sensor] = self._calculate_single_sensor(self, stride_event_list[sensor], sampling_rate_hz)
        return self

    def calculate(self: BaseType, stride_event_list: dict, sampling_rate_hz: float) -> BaseType:
        """Find temporal parameters of all strides after segmentation and detecting events for all sensors.

        Parameters
        ----------
        stride_event_list
            Gait events for each stride obtained from event detection
        sampling_rate_hz
            The sampling rate of the data signal.

        Returns
        -------
            self
                The class instance with temporal parameters populated in parameters_

        """
        if isinstance(stride_event_list, pd.DataFrame):  # this means single sensor
            self.parameters_ = self._calculate_single_sensor(stride_event_list, sampling_rate_hz)
        else:
            self.parameters_ = self._calculate_multiple_sensor(stride_event_list, sampling_rate_hz)
        return self


def _calc_time_stamp_(ic_event: float, sampling_rate_hz: float) -> float:
    return ic_event / sampling_rate_hz


def _calc_stride_time(ic_event: float, pre_ic_event: float, sampling_rate_hz: float) -> float:
    return (ic_event - pre_ic_event) / sampling_rate_hz


def _calc_swing_time(ic_event: float, tc_event: float, sampling_rate_hz: float) -> float:
    return (ic_event - tc_event) / sampling_rate_hz


def _calc_stance_time(stride_time: float, swing_time: float) -> float:
    return stride_time - swing_time
