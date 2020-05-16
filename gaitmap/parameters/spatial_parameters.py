"""Calculate spatial parameters algorithm by Kanzler et al. 2015 and Rampp et al. 2014."""
from typing import Union, Dict, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.spatial.transform import Rotation

from gaitmap.base import BaseType, BaseSpatialParameterCalculation
from gaitmap.parameters.temporal_parameters import _calc_stride_time
from gaitmap.utils.consts import GF_POS, GF_ORI
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
    set_correct_index,
)
from gaitmap.utils.rotations import find_angle_between_orientations, find_unsigned_3d_angle


class SpatialParameterCalculation(BaseSpatialParameterCalculation):
    """This class is responsible for calculating spatial parameters of strides based on [1]_ and [2]_.

    Attributes
    ----------
    parameters_
        Data frame containing spatial parameters for each stride in case of single sensor
        or dictionary of data frames in multi sensors.
        It has the same structure as the provided stride list
    parameters_pretty_
        The same as parameters_ but with column names including units.
    sole_angle_course_
        The sole angle of all strides over time.
        It has the same structure as the provided position list.


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
    Stride Length
        The stride length is calculated based on the pythagoras in the floor plane (x, y - plane)
    Gait Velocity
        The gait velocity is calculated by dividing the stride length by the stride time.
        Note, that the stride time is calculated from `pre_ic` to `ic` if a `min_vel` type stride list is provided.
        The stride is estimated based on the position at the `start` and the `end` of a stride.
        This means these two measures are calculated from different time periods which might lead to errors in
        certain edge cases.
    Arc Length
        The overall arc length is directly calculated from the position of the sensor by adding the absolute changes in
        position at every time point.
    Turning Angle
        The turning angle is calculated as the difference in orientation of the forward direction between the first
        and the last sample of each stride.
        Only the rotation around the z-axis (upwards) is considered.
        A turn to the left results in a positive and a turn to the right in a negative turning angle independent of
        the foot.
    IC/TC Angle and angle course
        All angles are calculated as the angle between the forward direction ([1, 0, 0] from the sensor frame
        transformed into the world frame, using the provided orientations) and the floor.
        The angle is positive if the vector is pointing upwards (i.e. the toe is higher than the heel) and negative
        if the angle is pointing downwards (i.e. the heel is higher than the toe), following the convetion in [1]_.
        The sole angle is assumed to be 0 during midstance.
        The IC and TC angles are simply the sole angles at the respective time points.


    .. [1] Kanzler, C. M., Barth, J., Rampp, A., Schlarb, H., Rott, F., Klucken, J., Eskofier, B. M. (2015, August).
       Inertial sensor based and shoe size independent gait analysis including heel and toe clearance estimation.
       In 2015 37th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)
       (pp. 5424-5427). IEEE.
    .. [2] Rampp, A., Barth, J., Schülein, S., Gaßmann, K. G., Klucken, J., & Eskofier, B. M. (2014).
       Inertial sensor-based stride parameter calculation from gait sequences in geriatric patients.
       IEEE transactions on biomedical engineering, 62(4), 1089-1097.

    Examples
    --------
    This method requires the output of a event detection method and a full trajectory reconstruction
    (orientation and position) as input.

    >>> stride_list = ...  # from event detection
    >>> positions = ...  # from position estimation
    >>> orientations = ...  # from orientation estimation
    >>> p = SpatialParameterCalculation()
    >>> p = p.calculate(stride_event_list=stride_list, positions=positions, orientations=orientations,
    ...                 sampling_rate_hz=204.8)
    >>> p.parameters_
    <Dataframe/dictionary with all the parameters>
    >>> p.parameters_pretty_
    <Dataframe/dictionary with all the parameters with units included in column names>

    See Also
    --------
    gaitmap.parameters.TemporalParameterCalculation: Calculate temporal parameters

    """

    parameters_: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    sole_angle_course_: PositionList

    stride_event_list: StrideList
    positions: PositionList
    orientations: OrientationList
    sampling_rate_hz: float

    @property
    def parameters_pretty_(self) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
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
            "stride_length": "stride length [m]",
            "gait_velocity": "gait velocity [m/s]",
            "ic_angle": "ic angle [deg]",
            "tc_angle": "tc angle [deg]",
            "turning_angle": "turning angle [deg]",
            "arc_length": "arc length [m]",
        }
        return parameters.rename(columns=pretty_columns)

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
            The class instance with spatial parameters populated in `self.parameters_`, `self.parameters_pretty_`

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
            self.parameters_, self.sole_angle_course_ = self._calculate_single_sensor(
                stride_event_list, positions, orientations, sampling_rate_hz
            )
        elif (
            is_multi_sensor_stride_list(stride_event_list, stride_type="min_vel")
            and is_multi_sensor_position_list(positions)
            and is_multi_sensor_orientation_list(orientations)
        ):
            self.parameters_, self.sole_angle_course_ = self._calculate_multiple_sensor(
                stride_event_list, positions, orientations, sampling_rate_hz
            )
        else:
            raise ValueError("The provided combinations of input types is not supported.")
        return self

    @staticmethod
    def _calculate_single_sensor(
        stride_event_list: SingleSensorStrideList,
        positions: SingleSensorPositionList,
        orientations: SingleSensorOrientationList,
        sampling_rate_hz: float,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Find spatial parameters of each stride in case of single sensor.

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
        sole_angle_course_
            The sole angle in the sagttial plane for each stride

        """
        positions = set_correct_index(positions, ["s_id", "sample"])[GF_POS]
        orientations = set_correct_index(orientations, ["s_id", "sample"])[GF_ORI]
        stride_event_list = set_correct_index(stride_event_list, ["s_id"])

        stride_length_ = _calc_stride_length(positions)
        gait_velocity_ = _calc_gait_velocity(
            stride_length_, _calc_stride_time(stride_event_list["ic"], stride_event_list["pre_ic"], sampling_rate_hz),
        )
        arc_length_ = _calc_arc_length(positions)
        turning_angle_ = _calc_turning_angle(orientations)

        angle_course_ = _compute_sole_angle_course(orientations)
        ic_relative = (stride_event_list["ic"] - stride_event_list["start"]).astype(int)
        tc_relative = (stride_event_list["tc"] - stride_event_list["start"]).astype(int)
        ic_angle_ = _get_angle_at_index(angle_course_, ic_relative)
        tc_angle_ = _get_angle_at_index(angle_course_, tc_relative)

        stride_parameter_dict = {
            "stride_length": stride_length_,
            "gait_velocity": gait_velocity_,
            "ic_angle": ic_angle_,
            "tc_angle": tc_angle_,
            "turning_angle": turning_angle_,
            "arc_length": arc_length_,
        }
        parameters_ = pd.DataFrame(stride_parameter_dict, index=stride_event_list.index)
        return parameters_, angle_course_

    def _calculate_multiple_sensor(
        self: BaseType,
        stride_event_list: MultiSensorStrideList,
        positions: MultiSensorPositionList,
        orientations: MultiSensorOrientationList,
        sampling_rate_hz: float,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
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
        sole_angle_course_
            The sole angle in the sagttial plane for each stride

        """
        parameters_ = {}
        sole_angle_course_ = {}
        for sensor in stride_event_list:
            parameters_[sensor], sole_angle_course_[sensor] = self._calculate_single_sensor(
                stride_event_list[sensor], positions[sensor], orientations[sensor], sampling_rate_hz
            )
        return parameters_, sole_angle_course_


def _calc_stride_length(positions: pd.DataFrame) -> pd.Series:
    start = positions.groupby(level="s_id").first()
    end = positions.groupby(level="s_id").last()
    stride_length = end - start
    stride_length = pd.Series(norm(stride_length[["pos_x", "pos_y"]], axis=1), index=stride_length.index)
    return stride_length


def _calc_gait_velocity(stride_length: pd.Series, stride_time: pd.Series) -> pd.Series:
    return stride_length / stride_time


def _get_angle_at_index(angle_course: np.array, index_per_stride: pd.Series) -> pd.Series:
    indexer = pd.MultiIndex.from_frame(index_per_stride.reset_index())
    return angle_course[indexer].reset_index(level=1, drop=True)


def _calc_turning_angle(orientations) -> pd.Series:
    start = orientations.groupby(level="s_id").first()
    end = orientations.groupby(level="s_id").last()
    angles = pd.Series(
        np.rad2deg(
            find_angle_between_orientations(
                Rotation.from_quat(end.to_numpy()), Rotation.from_quat(start.to_numpy()), [0, 0, 1]
            )
        ),
        index=start.index,
    )
    return angles


def _calc_arc_length(positions: pd.DataFrame) -> pd.Series:
    diff_per_sample = positions.groupby(level="s_id").diff().dropna()
    norm_per_sample = pd.Series(norm(diff_per_sample, axis=1), index=diff_per_sample.index)
    return norm_per_sample.groupby(level="s_id").sum()


def _compute_sole_angle_course(orientations: pd.DataFrame) -> pd.Series:
    """Find the angle between the "forward" vector and the ground.

    At every point in time we expect the sensor local x axis to point to the tip of the shoe.
    (It shouldn't matter if this is not perfectly true).
    Therefore, to find the sole angle of the shoe, we calculate the global orientation of the local x-axis and
    calculate its angle with the floor.

    # TODO: Is this different from calculating the angle only in the sagittal plane?
    # TODO: Linear dedrifting
    """
    forward = pd.DataFrame(
        Rotation.from_quat(orientations.to_numpy()).apply([1, 0, 0]), columns=list("xyz"), index=orientations.index
    )
    floor_angle = np.rad2deg(find_unsigned_3d_angle(forward.to_numpy(), np.array([0, 0, 1]))) - 90
    floor_angle = pd.Series(floor_angle, index=forward.index)
    return floor_angle
