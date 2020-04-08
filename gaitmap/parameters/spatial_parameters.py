"""Calculate spatial parameters algorithm."""
from typing import Union, Dict

import numpy as np
from numpy.linalg import norm

import pandas as pd

from gaitmap.parameters import temporal_parameters
from gaitmap.base import BaseType, BaseSpatialParameterCalculation
from gaitmap.utils.dataset_helper import (
    StrideList,
    MultiSensorStrideList,
    SingleSensorStrideList,
    is_single_sensor_stride_list,
    is_multi_sensor_stride_list,
    PositionList,
    SingleSensorPositionList,
    MultiSensorPositionList,
    is_single_sensor_position_list,
    is_multi_sensor_position_list,
    OrienationList,
    SingleSensorOrientationList,
    MultiSensorOrientationList,
    is_single_sensor_orientation_list,
    is_multi_sensor_orientation_list,
)


class SpatialParameterCalculation(BaseSpatialParameterCalculation):
    """This class is responsible for calculating spatial parameters of strides.

    Parameters
    ----------
    stride_event_list
        Gait events for each stride obtained from Rampp event detection.
    position
        position of the sensor at each time point as estimated by trajectory reconstruction.
    orientation
        orientation of the sensor at each time point as estimated by trajectory reconstruction.

    Attributes
    ----------
    parameters_
        Data frame containing spatial parameters for each stride in case of single sensor
        or dictionary of data frames in multi sensors.

    """

    parameters_: Union[pd.DataFrame, Dict[str, pd.DataFrame]]

    @staticmethod
    def _calculate_single_sensor(stride_event_list: SingleSensorStrideList, positions: SingleSensorPositionList,
                                 orientations: SingleSensorOrientationList,
                                 sampling_rate_hz: float) -> pd.DataFrame:
        """Find spatial parameters  of each stride in case of single sensor.

        Parameters
        ----------
        stride_event_list
            Gait events for each stride obtained from Rampp event detection.
        positions
            position of the sensor at each time point as estimated by trajectory reconstruction.
        orientations
            orientation of the sensor at each time point as estimated by trajectory reconstruction
        sampling_rate_hz
            The sampling rate of the data signal.

        Returns
        -------
        parameters_
            Data frame containing spatial parameters of single sensor

        """
        stride_length_ = []
        stride_id_ = stride_event_list["s_id"]
        position = positions.groupby('s_id')['position'].apply(list)
        for row in position.iteritems():
            stride_length_.append(_calc_stride_length(row[1]))
        gait_velocity_ = _calc_gait_velocity(stride_length_,
                                             temporal_parameters.calc_stride_time(stride_event_list["ic"],
                                                                                  stride_event_list["pre_ic"],
                                                                                  sampling_rate_hz))
        stride_parameter_dict = {
            "s_id": stride_id_,
            "stride_length": stride_length_,
            "gait_velocity": gait_velocity_,
        }
        parameters_ = pd.DataFrame(stride_parameter_dict)
        return parameters_

    def _calculate_multiple_sensor(
        self: BaseType, stride_event_list: MultiSensorStrideList, positions: MultiSensorPositionList,
            orientations: MultiSensorOrientationList, sampling_rate_hz: float
    ) -> Dict[str, pd.DataFrame]:
        """Find spatial parameters of each stride in case of multiple sensors.

        Parameters
        ----------
        stride_event_list
            Gait events for each stride obtained from Rampp event detection.
        positions
            position of each sensor at each time point as estimated by trajectory reconstruction.
        orientations
            orientation of each sensor at each time point as estimated by trajectory reconstruction
        sampling_rate_hz
            The sampling rate of the data signal.

        Returns
        -------
        parameters_
            Data frame containing spatial parameters of single sensor

        """
        parameters_ = {}
        for sensor in stride_event_list:
            parameters_[sensor] = self._calculate_single_sensor(stride_event_list[sensor], positions[sensor],
                                                                orientations[sensor], sampling_rate_hz)
        return parameters_

    def calculate(self: BaseType, stride_event_list: StrideList, positions: PositionList,
                  orientations: OrienationList, sampling_rate_hz: float) -> BaseType:
        """Find spatial parameters of all strides after segmentation and detecting events for all sensors.

        Parameters
        ----------
        stride_event_list
            Gait events for each stride obtained from event detection
        positions
            position of each sensor at each time point as estimated by trajectory reconstruction.
        orientations
            orientation of each sensor at each time point as estimated by trajectory reconstruction
        sampling_rate_hz
            The sampling rate of the data signal.

        Returns
        -------
        self
            The class instance with spatial parameters populated in parameters_

        """
        if is_single_sensor_stride_list(stride_event_list, stride_type="min_vel") and \
                is_single_sensor_position_list(positions) and \
                is_single_sensor_orientation_list(orientations): # this means single sensor
            self.parameters_ = self._calculate_single_sensor(stride_event_list, positions,
                                                             orientations, sampling_rate_hz)
        elif is_multi_sensor_stride_list(stride_event_list, stride_type="min_vel") and \
                is_multi_sensor_position_list(positions) and \
                is_multi_sensor_orientation_list(orientations):
            self.parameters_ = self._calculate_multiple_sensor(stride_event_list, positions,
                                                               orientations, sampling_rate_hz)
        else:
            raise ValueError("Stride list datatype is not supported.")
        return self


def _calc_stride_length(position: np.ndarray) -> float:
    position_end = position[len(position) - 1]
    return norm([position_end[0], position_end[2]])


def _calc_gait_velocity(stride_length: float, stride_time: float) -> float:
    return stride_length / stride_time
