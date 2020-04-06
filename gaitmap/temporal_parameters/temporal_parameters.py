import pandas as pd

from gaitmap.base import BaseType, BaseTemporalParameterCalculation
from gaitmap.event_detection.rampp_event_detection import RamppEventDetection


class TemporalParameterCalculation(BaseTemporalParameterCalculation):
    """ This class is responsible for calculating temporal parameters of strides.

    Parameters
    ----------
    gait_events
        gait events for each stride calculated by Rampp event detection.
    sampling_rate_hz
        The sampling rate of the data signal.
    parameters_
        Data frame containing temporal parameters for each stride from each sensor

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

    parameters_: pd.DataFrame  # Results are stored here
    stride_id_ = []
    time_stamp_ = []
    stride_time_ = []
    swing_time_ = []
    stance_time_ = []

    def __init__(self):
        pass

    @staticmethod
    def calculate(self: BaseType, gait_events: RamppEventDetection, sampling_rate_hz: float) -> BaseType:
        """Find temporal parameters in in strides after segmentation and detecting events of each stride.
        Parameters
        ----------
        gait_events
            Gait events obtained from event detection
        sampling_rate_hz
            The sampling rate of the data signal.

        Returns
        -------
            self
                The class instance with temporal parameters populated in parameters_

        """
        param = dict.fromkeys(["left_sensor", "right_sensor"])
        keys = ["stride_id", "timestamp", "stride_time", "swing_time", "stance_time"]
        for sensor in gait_events.data:
            for index, stride in gait_events.data[sensor].iterrows():
                self.stride_id_.append(stride["s_id"])
                self.time_stamp_.append(self._calc_timestamp(stride["ic"], sampling_rate_hz))
                self.stride_time_.append(self._calc_stride_time(stride["ic"], stride["pre_ic"], sampling_rate_hz))
                self.swing_time_.append(self._calc_swing_time(stride["ic"], stride["tc"], sampling_rate_hz))
                self.stance_time_.append(self.stride_time[index], self.swing_time[index])
            dic = dict(
                zip(keys, ([self.stride_id_, self.time_stamp_, self.stride_time_, self.swing_time_, self.stance_time_]))
            )
            param[sensor] = dic
        self.parameters_ = pd.DataFrame.from_dict(param)
        return self

    def _calc_time_stamp(self, ic_event: float, sampling_rate_hz: float) -> float:
        return ic_event / sampling_rate_hz

    def _calc_stride_time(self, ic_event: float, pre_ic_event: float, sampling_rate_hz: float) -> float:
        return (ic_event - pre_ic_event) / sampling_rate_hz

    def _calc_swing_time(self, ic_event: float, tc_event: float, sampling_rate_hz: float) -> float:
        return (ic_event - tc_event) / sampling_rate_hz

    def _calc_stance_time(self, stride_time: float, swing_time: float) -> float:
        return stride_time - swing_time
