"""Calculate temporal parameters algorithm."""
from typing import Union, Dict

import pandas as pd

from gaitmap.base import BaseType, BaseTemporalParameterCalculation
from gaitmap.utils.dataset_helper import (
    StrideList,
    MultiSensorStrideList,
    SingleSensorStrideList,
    is_single_sensor_stride_list,
    is_multi_sensor_stride_list,
)


class TemporalParameterCalculation(BaseTemporalParameterCalculation):
    """This class is responsible for calculating temporal parameters of strides.

    Attributes
    ----------
    parameters_
        Data frame containing temporal parameters for each stride in case of single sensor
        or dictionary of data frames in multi sensors.

    Other Parameters
    ----------------
    stride_event_list
            Gait events for each stride obtained from Rampp event detection.
    sampling_rate_hz
        The sampling rate of the data signal.


    """

    parameters_: Union[pd.DataFrame, Dict[str, pd.DataFrame]]

    sampling_rate_hz: float
    stride_event_list: StrideList

    def calculate(self: BaseType, stride_event_list: StrideList, sampling_rate_hz: float) -> BaseType:
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
            The class instance with temporal parameters populated in `self.parameters_`

        """
        self.sampling_rate_hz = sampling_rate_hz
        self.stride_event_list = stride_event_list
        if is_single_sensor_stride_list(stride_event_list, stride_type="min_vel"):  # this means single sensor
            self.parameters_ = self._calculate_single_sensor(stride_event_list, sampling_rate_hz)
        elif is_multi_sensor_stride_list(stride_event_list, stride_type="min_vel"):
            self.parameters_ = self._calculate_multiple_sensor(stride_event_list, sampling_rate_hz)
        else:
            raise ValueError("Stride list datatype is not supported.")
        return self

    @staticmethod
    def _calculate_single_sensor(stride_event_list: SingleSensorStrideList, sampling_rate_hz: float) -> pd.DataFrame:
        """Find temporal parameters  of each stride in case of single sensor.

        Parameters
        ----------
        stride_event_list
            Gait events for each stride obtained from event detection
        sampling_rate_hz
            The sampling rate of the data signal.

        Returns
        -------
        parameters_
            Data frame containing temporal parameters of single sensor

        """
        stride_id_ = stride_event_list["s_id"]
        time_stamp_ = _calc_time_stamp_(stride_event_list["ic"], sampling_rate_hz)
        stride_time_ = calc_stride_time(stride_event_list["ic"], stride_event_list["pre_ic"], sampling_rate_hz)
        swing_time_ = _calc_swing_time(stride_event_list["ic"], stride_event_list["tc"], sampling_rate_hz)
        stance_time_ = [stride_time - swing_time for stride_time, swing_time in zip(stride_time_, swing_time_)]
        stride_parameter_dict = {
            "stride_id": stride_id_,
            "time_stamp": time_stamp_,
            "stride_time": stride_time_,
            "swing_time": swing_time_,
            "stance_time": stance_time_,
        }
        parameters_ = pd.DataFrame(stride_parameter_dict)
        return parameters_

    def _calculate_multiple_sensor(
        self: BaseType, stride_event_list: MultiSensorStrideList, sampling_rate_hz: float
    ) -> Dict[str, pd.DataFrame]:
        """Find temporal parameters of each stride in case of multiple sensors.

        Parameters
        ----------
        stride_event_list
            Gait events for each stride obtained from event detection
        sampling_rate_hz
            The sampling rate of the data signal.

        Returns
        -------
        parameters_
            Dictionary of temporal parameters for each sensor

        """
        parameters_ = {}
        for sensor in stride_event_list:
            parameters_[sensor] = self._calculate_single_sensor(stride_event_list[sensor], sampling_rate_hz)
        return parameters_


def _calc_time_stamp_(ic_event: float, sampling_rate_hz: float) -> float:
    return ic_event / sampling_rate_hz


def calc_stride_time(ic_event: pd.Series, pre_ic_event: pd.Series, sampling_rate_hz: float) -> pd.Series:
    """Find stride time.

    Parameters
    ----------
    ic_event
        Initial contact event from event detection
    pre_ic_event
        Previous Initial contact event from event detection
    sampling_rate_hz
        The sampling rate of the data signal.

    Returns
    -------
        Stride time

    """
    return (ic_event - pre_ic_event) / sampling_rate_hz


def _calc_swing_time(ic_event: float, tc_event: float, sampling_rate_hz: float) -> float:
    return (ic_event - tc_event) / sampling_rate_hz


def _calc_stance_time(stride_time: float, swing_time: float) -> float:
    return stride_time - swing_time
