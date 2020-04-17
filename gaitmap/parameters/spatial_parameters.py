"""Calculate spatial parameters algorithm by Kanzler et al. 2015."""
import math
from typing import Union, Dict

import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.spatial.transform import Rotation

from gaitmap.base import BaseType, BaseSpatialParameterCalculation
from gaitmap.parameters.temporal_parameters import calc_stride_time
from gaitmap.utils import vector_math
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
    OrientationList,
    SingleSensorOrientationList,
    MultiSensorOrientationList,
    is_single_sensor_orientation_list,
    is_multi_sensor_orientation_list,
)


class SpatialParameterCalculation(BaseSpatialParameterCalculation):
    """This class is responsible for calculating spatial parameters of strides.

    Attributes
    ----------
    parameters_
        Data frame containing spatial parameters for each stride in case of single sensor
        or dictionary of data frames in multi sensors.

    Other Parameters
    ----------------
    stride_event_list
        Gait events for each stride obtained from event detection.
    positions
        position of the sensor at each time point as estimated by trajectory reconstruction.
    orientations
        orientation of the sensor at each time point as estimated by trajectory reconstruction.
    sampling_rate_hz
        The sampling rate of the data signal.

    Notes
    -----
    .. [1] Kanzler, C. M., Barth, J., Rampp, A., Schlarb, H., Rott, F., Klucken, J., Eskofier, B. M. (2015, August).
       Inertial sensor based and shoe size independent gait analysis including heel and toe clearance estimation.
       In 2015 37th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)
       (pp. 5424-5427). IEEE.

    """

    parameters_: Union[pd.DataFrame, Dict[str, pd.DataFrame]]

    stride_event_list: StrideList
    positions: PositionList
    orientations: OrientationList
    sampling_rate_hz: float

    def calculate(
        self: BaseType,
        stride_event_list: StrideList,
        positions: PositionList,
        orientations: OrientationList,
        sampling_rate_hz: float,
    ) -> BaseType:
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
        self.stride_event_list = stride_event_list
        self.positions = positions
        self.orientations = orientations
        self.sampling_rate_hz = sampling_rate_hz
        if (
            is_single_sensor_stride_list(stride_event_list, stride_type="min_vel")
            and is_single_sensor_position_list(positions)
            and is_single_sensor_orientation_list(orientations)
        ):
            self.parameters_ = self._calculate_single_sensor(
                stride_event_list, positions, orientations, sampling_rate_hz
            )
        elif (
            is_multi_sensor_stride_list(stride_event_list, stride_type="min_vel")
            and is_multi_sensor_position_list(positions)
            and is_multi_sensor_orientation_list(orientations)
        ):
            self.parameters_ = self._calculate_multiple_sensor(
                stride_event_list, positions, orientations, sampling_rate_hz
            )
        else:
            raise ValueError("THe provided combinations of input types is not supported.")
        return self

    @staticmethod
    def _calculate_single_sensor(
        stride_event_list: SingleSensorStrideList,
        positions: SingleSensorPositionList,
        orientations: SingleSensorOrientationList,
        sampling_rate_hz: float,
    ) -> pd.DataFrame:
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
        positions = positions.set_index(('s_id', 'sample'))
        stride_length_ = _calc_stride_length(positions)
        gait_velocity_ = _calc_gait_velocity(
            stride_length_, calc_stride_time(stride_event_list["ic"], stride_event_list["pre_ic"], sampling_rate_hz),
        )
        angle_course_ = _compute_sagittal_angle_course(
            orientation_x[1], orientation_y[1], orientation_z[1], orientation_w[1]
        )
        ic_relative = stride_event_list["ic"] - stride_event_list["start"]
        tc_relative = stride_event_list["tc"] - stride_event_list["start"]
        ic_clearance_ = _calc_ic_clearance(pos_y[1], angle_course_, ic_relative[1])
        tc_clearance_ = _calc_tc_clearance(pos_y[1], angle_course_, tc_relative[1])
        ic_angle_ = _calc_ic_angle(angle_course_, ic_relative[1])
        tc_angle_ = _calc_tc_angle(angle_course_, tc_relative[1])
        turning_angle_ = _calc_turning_angle(orientation_x[1], orientation_y[1], orientation_z[1], orientation_w[1])
        arc_length_ = _calc_arc_length(pos_x[1], pos_y[1], pos_z[1])

        stride_parameter_dict = {
            "s_id": stride_id_,
            "stride_length": stride_length_,
            "gait_velocity": gait_velocity_,
            "ic_clearance": ic_clearance_,
            "tc_clearance": tc_clearance_,
            "ic_angle": ic_angle_,
            "tc_angle": tc_angle_,
            "turning_angle": turning_angle_,
            "arc_length": arc_length_,
        }
        parameters_ = pd.DataFrame(stride_parameter_dict)
        return parameters_

    def _calculate_multiple_sensor(
        self: BaseType,
        stride_event_list: MultiSensorStrideList,
        positions: MultiSensorPositionList,
        orientations: MultiSensorOrientationList,
        sampling_rate_hz: float,
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
            parameters_[sensor] = self._calculate_single_sensor(
                stride_event_list[sensor], positions[sensor], orientations[sensor], sampling_rate_hz
            )
        return parameters_


def _calc_stride_length(positions: pd.DataFrame) -> pd.Series:
    stride_length = positions.groupby(level='s_id').nth([0, -1]).groupby(level='s_id').diff().dropna(axis=0)
    stride_length = pd.Series(norm(stride_length[['pos_x', 'pos_y']], axis=1), index=stride_length.index)
    return stride_length


def _calc_gait_velocity(stride_length: pd.Series, stride_time: pd.Series) -> pd.Series:
    return stride_length / stride_time


def _calc_ic_clearance(pos_y: np.array, angle_course: np.array, ic_relative: int) -> np.array:
    sensor_lift_ic = pos_y[int(ic_relative)]
    l_ic = sensor_lift_ic / math.sin(angle_course[int(ic_relative)])
    ic_clearance = []
    sensor_clearance = pos_y
    for i, _ in enumerate(pos_y):
        sgn = np.sign(angle_course[i])
        delta_ic = sgn * l_ic * math.sin(angle_course[i])
        ic_clearance.append(-sensor_clearance[i] + sgn * delta_ic)
    return ic_clearance


def _calc_tc_clearance(pos_y: np.array, angle_course: np.array, tc_relative: int) -> np.array:
    sensor_lift_tc = pos_y[int(tc_relative)]
    l_tc = sensor_lift_tc / math.sin(angle_course[int(tc_relative)])
    tc_clearance = []
    sensor_clearance = pos_y
    for i, _ in enumerate(pos_y):
        sgn = np.sign(angle_course[i])
        delta_tc = sgn * l_tc * math.sin(angle_course[i])
        tc_clearance.append(-sensor_clearance[i] + sgn * delta_tc)
    return tc_clearance


def _calc_ic_angle(angle_course: np.array, ic_relative: int) -> float:
    return -np.rad2deg(angle_course[int(ic_relative)])


def _calc_tc_angle(angle_course: np.array, tc_relative: int) -> float:
    return -np.rad2deg(angle_course[int(tc_relative)])


def _calc_turning_angle(
    orientation_x: np.array, orientation_y: np.array, orientation_z: np.array, orientation_w: np.array
) -> float:
    orientation_turn = vector_math.inner_product(
        np.array([orientation_x[0], orientation_y[0], orientation_z[0], orientation_w[0]]),
        vector_math.inverse(
            np.array(
                [
                    orientation_x[len(orientation_x) - 1],
                    orientation_y[len(orientation_y) - 1],
                    orientation_z[len(orientation_z) - 1],
                    orientation_w[len(orientation_w) - 1],
                ]
            )
        ),
    )
    return np.rad2deg(Rotation.from_quat(orientation_turn).as_euler("zyx", degrees=True)[1])


def _calc_arc_length(pos_x: np.array, pos_y: np.array, pos_z: np.array) -> float:
    arc_length = 0
    for index in pos_x[: len(pos_x) - 1]:
        arc_length += norm(
            np.array(
                [pos_x[index + 1] - pos_x[index], pos_y[index + 1] - pos_y[index], pos_z[index + 1] - pos_z[index]]
            )
        )
    return arc_length


def _compute_sagittal_angle_course(qx: np.array, qy: np.array, qz: np.array, qw: np.array) -> np.array:
    angle_course = []
    for i, _ in enumerate(qx):
        orientation_ms = vector_math.inner_product(
            np.array([qx[i], qy[i], qz[i], qw[i]]), np.array([qx[0], qy[0], qz[0], qw[0]])
        )
        angle_course.append(Rotation.from_quat(orientation_ms).as_euler("zyx", degrees=True)[2])
    return angle_course
