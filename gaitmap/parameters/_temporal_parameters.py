"""Calculate temporal parameters algorithm."""

from typing import Literal, TypeVar, Union

import pandas as pd

from gaitmap.base import BaseTemporalParameterCalculation
from gaitmap.utils.consts import SL_INDEX
from gaitmap.utils.datatype_helper import (
    SingleSensorStrideList,
    StrideList,
    is_stride_list,
    set_correct_index,
)

Self = TypeVar("Self", bound="TemporalParameterCalculation")


class TemporalParameterCalculation(BaseTemporalParameterCalculation):
    """Calculat temporal parameters of strides based on detected gait events.

    For details on the individual parameters see the Notes section.
    Calculations are based on [1]_.

    Parameters
    ----------
    expected_stride_type
        The expected stride type of the stride list.
        This changes how the temporal parameters are calculated.
        This can either be "min_vel" or "ic".
        "min_vel" stride lists are the typical output from Gaitmap event detection methods.
        However, for other systems (e.g. mocap systems) strides might be defined from one ic to the next ic.
        In this case the expected_stride_type should be "ic".

    Attributes
    ----------
    parameters_
        Data frame containing temporal parameters for each stride in case of single sensor
        or dictionary of data frames in multi sensors.
    parameters_pretty_
        The same as parameters_ but with column names including units.

    Other Parameters
    ----------------
    stride_event_list
            Gait events for each stride obtained from Rampp event detection as type `min_vel`-stride list.
    sampling_rate_hz
        The sampling rate of the data signal.

    Notes
    -----
    stride_time [s]
        The stride time is the duration of the stride calculated based on the ic events of the stride.
        For a `min_vel`-stride the stride time is calculated by subtracting "pre_ic" from "ic".
        For a `ic`-stride the stride time is calculated by subtracting "start"/"ic" from "end".
    swing_time [s]
        The swing time is the time from the tc to the next ic.
        For a `min_vel`-stride this is the time between "tc" and "ic"
        For a `ic`-stride this is the time between "tc" and "end".
    stance_time [s]
        The stance time is the time the foot is on the ground.
        Hence, it is the time from a ic to the next tc.
        For both stride types this is calculated as stride_time - swing_time.

    .. [1] A. Rampp, J. Barth, S. Schuelein, K.-G. Gassmann, J. Klucken, and B. M. Eskofier, “Inertial Sensor-Based
       Stride Parameter Calculation From Gait Sequences in Geriatric Patients,” IEEE Transactions on Biomedical
       Engineering, vol. 62, no. 4, pp. 1089-1097, Apr. 2015. [Online].
       Available: http://ieeexplore.ieee.org/document/6949634/

    Examples
    --------
    This method requires the output of a event detection method as input.

    >>> stride_list = ...  #  from event detection
    >>> temporal_paras = TemporalParameterCalculation()
    >>> temporal_paras = temporal_paras.calculate(stride_event_list=stride_list, sampling_rate_hz=204.8)
    >>> temporal_paras.parameters_
    <Dataframe/dictionary with all the parameters>
    >>> temporal_paras.parameters_pretty_
    <Dataframe/dictionary with all the parameters with units included in column names>

    See Also
    --------
    gaitmap.parameters.SpatialParameterCalculation: Calculate spatial parameters

    """

    expected_stride_type: Literal["min_vel", "ic"]

    parameters_: Union[pd.DataFrame, dict[str, pd.DataFrame]]

    sampling_rate_hz: float
    stride_event_list: StrideList

    def __init__(self, expected_stride_type: Literal["min_vel", "ic"] = "min_vel") -> None:
        self.expected_stride_type = expected_stride_type

    @property
    def parameters_pretty_(self) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
        """Return parameters with column names indicating units."""
        if isinstance(self.parameters_, dict):
            parameters_ = {}
            for sensor in self.parameters_:
                parameters_[sensor] = self._rename_columns(self.parameters_[sensor])
            return parameters_
        return self._rename_columns(self.parameters_)

    @staticmethod
    def _rename_columns(parameters: pd.DataFrame) -> pd.DataFrame:
        pretty_columns = {
            "stride_time": "stride time [s]",
            "swing_time": "swing time [s]",
            "stance_time": "stance time [s]",
        }
        renamed_paras = parameters.rename(columns=pretty_columns)
        renamed_paras.index.name = "stride id"
        return renamed_paras

    def calculate(self: Self, stride_event_list: StrideList, sampling_rate_hz: float) -> Self:
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
            The class instance with temporal parameters populated in `self.parameters_`, `self.parameters_pretty_`

        """
        self.sampling_rate_hz = sampling_rate_hz
        self.stride_event_list = stride_event_list
        stride_list_type = is_stride_list(stride_event_list, stride_type=self.expected_stride_type)
        if stride_list_type == "single":  # this means single sensor
            self.parameters_ = self._calculate_single_sensor(stride_event_list, sampling_rate_hz)
        else:
            self.parameters_ = {
                sensor: self._calculate_single_sensor(stride_event_list[sensor], sampling_rate_hz)
                for sensor in stride_event_list
            }
        return self

    def _calculate_single_sensor(
        self, stride_event_list: SingleSensorStrideList, sampling_rate_hz: float
    ) -> pd.DataFrame:
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
        stride_event_list = set_correct_index(stride_event_list, SL_INDEX)

        stride_time_start_col, stride_time_end_col = _get_stride_time_cols(self.expected_stride_type)
        stride_time = (
            stride_event_list[stride_time_end_col] - stride_event_list[stride_time_start_col]
        ) / sampling_rate_hz

        swing_time_start_col, swing_time_end_col = _get_swing_time_cols(self.expected_stride_type)
        swing_time = (
            stride_event_list[swing_time_end_col] - stride_event_list[swing_time_start_col]
        ) / sampling_rate_hz
        stance_time = stride_time - swing_time
        stride_parameter_dict = {
            "stride_time": stride_time,
            "swing_time": swing_time,
            "stance_time": stance_time,
        }
        parameters_ = pd.DataFrame(stride_parameter_dict, index=stride_event_list.index)
        return parameters_


def _get_stride_time_cols(stride_type: Literal["min_vel", "ic"]) -> tuple[str, str]:
    if stride_type == "min_vel":
        return "pre_ic", "ic"
    if stride_type == "ic":
        return "start", "end"
    raise ValueError("stride_type should be either 'min_vel' or 'ic'")


def _get_swing_time_cols(stride_type: Literal["min_vel", "ic"]) -> tuple[str, str]:
    if stride_type == "min_vel":
        return "tc", "ic"
    if stride_type == "ic":
        return "tc", "end"
    raise ValueError("stride_type should be either 'min_vel' or 'ic'")
