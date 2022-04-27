"""Calculate spatial parameters algorithm by Kanzler et al. 2015 and Rampp et al. 2014."""
from typing import Dict, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.spatial.transform import Rotation

from gaitmap.base import BaseSpatialParameterCalculation
from gaitmap.parameters.temporal_parameters import _calc_stride_time
from gaitmap.utils._types import _Hashable
from gaitmap.utils.consts import GF_INDEX, GF_ORI, GF_POS, SL_INDEX
from gaitmap.utils.datatype_helper import (
    MultiSensorOrientationList,
    MultiSensorPositionList,
    MultiSensorStrideList,
    OrientationList,
    PositionList,
    SingleSensorOrientationList,
    SingleSensorPositionList,
    SingleSensorStrideList,
    StrideList,
    is_orientation_list,
    is_position_list,
    is_stride_list,
    set_correct_index,
)
from gaitmap.utils.exceptions import ValidationError
from gaitmap.utils.rotations import find_angle_between_orientations, find_unsigned_3d_angle

Self = TypeVar("Self", bound="SpatialParameterCalculation")


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
        Gait events for each stride obtained from event detection as type `min_vel`-stride list.
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
    Max. Sensor Lift
        The maximal relative height (relative to the height at midstance) the sensor reaches during the stride.
        Note that this is not equivalent to the actual foot lift/toe clearance.
        These values can be estimated, if the postion of the sensor on the foot is known.
    Max. Lateral Excursion
        The maximal lateral distance between the foot and an imaginary straight line spanning from the start to the
        end position of each stride.
        This indicates "how far outwards" a subject moves the foot during the swing phase.
        Note, that this parameter is only meaningfull for straight strides.
        It is further assumed that the inward/outward rotation of the foot is small in comparison the the actual
        lateral swing of the leg.

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
    >>> spatial_paras = SpatialParameterCalculation()
    >>> spatial_paras = spatial_paras.calculate(
    ...                               stride_event_list=stride_list,
    ...                               positions=positions,
    ...                               orientations=orientations,
    ...                               sampling_rate_hz=204.8
    ...                  )
    >>> spatial_paras.parameters_
    <Dataframe/dictionary with all the parameters>
    >>> spatial_paras.parameters_pretty_
    <Dataframe/dictionary with all the parameters with units included in column names>

    See Also
    --------
    gaitmap.parameters.TemporalParameterCalculation: Calculate temporal parameters

    """

    parameters_: Union[pd.DataFrame, Dict[_Hashable, pd.DataFrame]]
    sole_angle_course_: PositionList

    stride_event_list: StrideList
    positions: PositionList
    orientations: OrientationList
    sampling_rate_hz: float

    @property
    def parameters_pretty_(self) -> Union[pd.DataFrame, Dict[_Hashable, pd.DataFrame]]:
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
            "max_sensor_lift": "max. sensor lift [m]",
            "max_lateral_excursion": "max. lateral excursion [m]",
        }
        renamed_paras = parameters.rename(columns=pretty_columns)
        renamed_paras.index.name = "stride id"
        return renamed_paras

    def calculate(
        self: Self,
        stride_event_list: StrideList,
        positions: PositionList,
        orientations: OrientationList,
        sampling_rate_hz: float,
    ) -> Self:
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
        stride_list_type = is_stride_list(stride_event_list, stride_type="min_vel")
        position_list_type = is_position_list(positions, position_list_type="stride")
        orientation_list_type = is_orientation_list(orientations, orientation_list_type="stride")
        if not stride_list_type == position_list_type == orientation_list_type:
            raise ValidationError(
                "The provided stride list, the positions, and the orientations should all either "
                "be all single or all multi sensor objects."
                "However, the provided stride list is {} sensor, the positions {} sensor and the "
                "orientations {} sensor.".format(stride_list_type, position_list_type, orientation_list_type)
            )
        if stride_list_type == "single":
            self.parameters_, self.sole_angle_course_ = self._calculate_single_sensor(
                stride_event_list, positions, orientations, sampling_rate_hz
            )
        else:
            self.parameters_, self.sole_angle_course_ = self._calculate_multiple_sensor(
                stride_event_list, positions, orientations, sampling_rate_hz
            )
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
            Gait events for each stride obtained from event detection.
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
        positions = set_correct_index(positions, GF_INDEX)[GF_POS]
        orientations = set_correct_index(orientations, GF_INDEX)[GF_ORI]
        stride_event_list = set_correct_index(stride_event_list, SL_INDEX)

        stride_length_ = _calc_stride_length(positions)
        gait_velocity_ = _calc_gait_velocity(
            stride_length_, _calc_stride_time(stride_event_list["ic"], stride_event_list["pre_ic"], sampling_rate_hz)
        )
        arc_length_ = _calc_arc_length(positions)
        turning_angle_ = _calc_turning_angle(orientations)
        max_sensor_lift_ = _calc_max_sensor_lift(positions)
        max_lateral_excursion_ = _calc_max_lateral_excursion(positions)

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
            "max_sensor_lift": max_sensor_lift_,
            "max_lateral_excursion": max_lateral_excursion_,
        }
        parameters_ = pd.DataFrame(stride_parameter_dict, index=stride_event_list.index)
        return parameters_, angle_course_

    def _calculate_multiple_sensor(
        self,
        stride_event_list: MultiSensorStrideList,
        positions: MultiSensorPositionList,
        orientations: MultiSensorOrientationList,
        sampling_rate_hz: float,
    ) -> Tuple[Dict[_Hashable, pd.DataFrame], Dict[_Hashable, pd.Series]]:
        """Find spatial parameters of each stride in case of multiple sensors.

        Parameters
        ----------
        stride_event_list
            Gait events for each stride obtained from event detection.
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
            The sole angle in the sagittal plane (around the "ml" axis) for each stride

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


def _get_angle_at_index(angle_course: np.ndarray, index_per_stride: pd.Series) -> pd.Series:
    indexer = pd.MultiIndex.from_frame(index_per_stride.reset_index())
    return angle_course[indexer].reset_index(level=1, drop=True)


def _calc_turning_angle(orientations: pd.DataFrame) -> pd.Series:

    # We need to handle the case of empty orientations df manually
    if orientations.empty:
        angles = pd.Series()
        angles.index.name = "s_id"
        return angles

    start = orientations.groupby(level="s_id").first()
    end = orientations.groupby(level="s_id").last()
    angles = pd.Series(
        np.rad2deg(
            find_angle_between_orientations(
                Rotation.from_quat(end.to_numpy()), Rotation.from_quat(start.to_numpy()), np.asarray([0, 0, 1])
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
    # We need to handle the case of empty orientations df manually
    if orientations.empty:
        floor_angle = pd.Series()
        floor_angle.index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=["s_id", "sample"])
        return floor_angle

    forward = pd.DataFrame(
        Rotation.from_quat(orientations.to_numpy()).apply([1, 0, 0]), columns=list("xyz"), index=orientations.index
    )
    floor_angle = np.rad2deg(find_unsigned_3d_angle(forward.to_numpy(), np.array([0, 0, 1]))) - 90
    floor_angle = pd.Series(floor_angle, index=forward.index)
    return floor_angle


def _calc_max_sensor_lift(positions: SingleSensorPositionList) -> pd.Series:
    return positions["pos_z"].groupby(level="s_id").max()


def _calc_max_lateral_excursion(positions: SingleSensorPositionList) -> pd.Series:
    """Calculate the maximal lateral deviation from a straight line going from start pos to end pos of a stride.

    x1 = (x1,y1), x2 = (x2,y2) define the line
    x = (x0,y0) is the point the distance is computed to
    d =  abs((x2-x1)(y1-y0) - (x1-x0)(y2-y1))/sqrt((x2-x1)**2+(y2-y1)**2)
      =  abs((x2-x1)(y1-y0) - (x1-x0)(y2-y1))/stride_length

    """
    start = positions.groupby(level="s_id").first()
    end = positions.groupby(level="s_id").last()
    stride_length = _calc_stride_length(positions)

    def _calc_per_stride(start, end, length, data):
        excursion = (
            (end["pos_x"] - start["pos_x"]) * (start["pos_y"] - data["pos_y"])
            - (start["pos_x"] - data["pos_x"]) * (end["pos_y"] - start["pos_y"])
        ).abs()
        max_excursion = excursion.max()
        return max_excursion / length

    max_lat_excursion = positions.groupby(level="s_id").apply(
        lambda x: _calc_per_stride(start.loc[x.name], end.loc[x.name], stride_length.loc[x.name], x)
    )
    return max_lat_excursion
