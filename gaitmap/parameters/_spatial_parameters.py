"""Calculate spatial parameters algorithm by Kanzler et al. 2015 and Rampp et al. 2014."""

import warnings
from collections.abc import Sequence
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
from typing_extensions import Self

from gaitmap.base import BaseSpatialParameterCalculation
from gaitmap.parameters._temporal_parameters import _get_stride_time_cols
from gaitmap.utils._algo_helper import invert_result_dictionary, set_params_from_dict
from gaitmap.utils._types import _Hashable
from gaitmap.utils.consts import GF_INDEX, GF_ORI, GF_POS, SL_INDEX
from gaitmap.utils.datatype_helper import (
    OrientationList,
    PositionList,
    SingleSensorOrientationList,
    SingleSensorPositionList,
    SingleSensorStrideList,
    StrideList,
    get_multi_sensor_names,
    is_orientation_list,
    is_position_list,
    is_stride_list,
    set_correct_index,
)
from gaitmap.utils.exceptions import ValidationError
from gaitmap.utils.rotations import find_angle_between_orientations, find_unsigned_3d_angle

ParamterNames = Literal[
    "stride_length",
    "gait_velocity",
    "max_orientation_change",
    "ic_angle",
    "tc_angle",
    "turning_angle",
    "arc_length",
    "max_sensor_lift",
    "max_lateral_excursion",
]


class SpatialParameterCalculation(BaseSpatialParameterCalculation):
    """Calculating spatial parameters of strides based on extracted gait events and foot trajectories.

    Calculations are based on [1]_ and [2]_.

    Parameters
    ----------
    calculate_only
        List of parameters to be calculated. If None, all parameters are calculated.
        Even, if all parameters are calculated, we will check for the existence of the required data.
    expected_stride_type
        The expected stride type of the stride list.
        This can either be "min_vel" or "ic".
        This effects how the events are used to calculate the stride time required for the gait velocity calculation.
        For more details see :class:`~gaitmap.parameters.TemporalParameterCalculation`.

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
    Max. orientation change
        The maximum change of angle in the sagittal plane. It is similar to the range of motion, however, the measured
        parameter is value effected by other joints such as knee and hip as well.
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
    This method requires the output of an event detection method and a full trajectory reconstruction
    (orientation and position) as input.

    >>> stride_list = ...  # from event detection
    >>> positions = ...  # from position estimation
    >>> orientations = ...  # from orientation estimation
    >>> spatial_paras = SpatialParameterCalculation()
    >>> spatial_paras = spatial_paras.calculate(
    ...     stride_event_list=stride_list, positions=positions, orientations=orientations, sampling_rate_hz=204.8
    ... )
    >>> spatial_paras.parameters_
    <Dataframe/dictionary with all the parameters>
    >>> spatial_paras.parameters_pretty_
    <Dataframe/dictionary with all the parameters with units included in column names>

    See Also
    --------
    gaitmap.parameters.TemporalParameterCalculation: Calculate temporal parameters

    """

    calculate_only: Optional[Sequence[ParamterNames]]
    expected_stride_type: Literal["min_vel", "ic"]

    parameters_: Union[pd.DataFrame, dict[_Hashable, pd.DataFrame]]
    sole_angle_course_: Optional[Union[pd.Series, dict[_Hashable, pd.Series]]]

    stride_event_list: StrideList
    positions: Optional[PositionList]
    orientations: Optional[OrientationList]
    sampling_rate_hz: float

    def __init__(
        self,
        calculate_only: Optional[Sequence[ParamterNames]] = None,
        expected_stride_type: Literal["min_vel", "ic"] = "min_vel",
    ) -> None:
        self.calculate_only = calculate_only
        self.expected_stride_type = expected_stride_type

    @property
    def parameters_pretty_(self) -> Union[pd.DataFrame, dict[_Hashable, pd.DataFrame]]:
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
            "max_orientation_change": "max. angle change [deg]",
            "ic_angle": "ic angle [deg]",
            "tc_angle": "tc angle [deg]",
            "turning_angle": "turning angle [deg]",
            "arc_length": "arc length [m]",
            "max_sensor_lift": "max. sensor lift [m]",
            "max_lateral_excursion": "max. lateral excursion [m]",
        }
        renamed_paras = parameters.rename(columns=pretty_columns, errors="ignore")
        renamed_paras.index.name = "stride id"
        return renamed_paras

    def calculate(
        self,
        stride_event_list: StrideList,
        positions: Optional[PositionList],
        orientations: Optional[OrientationList],
        sampling_rate_hz: float,
    ) -> Self:
        """Find spatial parameters of all strides after segmentation and detecting events for all sensors.

        Parameters
        ----------
        stride_event_list
            Gait events for each stride obtained from event detection
        positions
            Position of each sensor at each time point as estimated by trajectory reconstruction.
            Can be set to `None` if you are only interested in orientation based features.
        orientations
            Orientation of each sensor at each time point as estimated by trajectory reconstruction
            Can be set to `None` if you are only interested in position based features.
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
        stride_list_type = is_stride_list(
            stride_event_list, stride_type=self.expected_stride_type, check_additional_cols=False
        )
        if positions is None and orientations is None:
            raise ValidationError(
                "Either positions or orientations should be provided for spatial parameter calculation. "
                "Otherwise, no spatial parameters cannot be calculated."
            )
        if positions is not None:
            position_list_type = is_position_list(positions, position_list_type="stride")
            if position_list_type != stride_list_type:
                raise ValidationError(
                    f"The provided stride list is of type {stride_list_type} sensor but the provided positions are of "
                    f"type {position_list_type} sensor. "
                    "The stride list and the positions should be of the same type."
                )
        if orientations is not None:
            orientation_list_type = is_orientation_list(orientations, orientation_list_type="stride")
            if orientation_list_type != stride_list_type:
                raise ValidationError(
                    f"The provided stride list is of type {stride_list_type} sensor but the provided orientations are "
                    "of type {orientation_list_type} sensor. "
                    "The stride list and the positions should be of the same type."
                )

        if stride_list_type == "single":
            set_params_from_dict(
                self,
                self._calculate_single_sensor(stride_event_list, positions, orientations, sampling_rate_hz),
                result_formatting=True,
            )
        else:
            set_params_from_dict(
                self,
                invert_result_dictionary(
                    {
                        sensor: self._calculate_single_sensor(
                            stride_event_list[sensor],
                            positions[sensor] if positions is not None else None,
                            orientations[sensor] if orientations is not None else None,
                            sampling_rate_hz,
                        )
                        for sensor in get_multi_sensor_names(stride_event_list)
                    }
                ),
                result_formatting=True,
            )
        return self

    def _calculate_single_sensor(
        self,
        stride_event_list: SingleSensorStrideList,
        positions: SingleSensorPositionList,
        orientations: SingleSensorOrientationList,
        sampling_rate_hz: float,
    ):
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
            The sole angle in the sagittal plane for each stride

        """
        stride_event_list = set_correct_index(stride_event_list, SL_INDEX)

        stride_parameter_dict = {}

        if positions is not None:
            stride_parameter_dict = {
                **stride_parameter_dict,
                **self._traj_based_parameters(positions, stride_event_list, sampling_rate_hz),
            }

        if orientations is not None:
            orientations = set_correct_index(orientations, GF_INDEX)[GF_ORI]
            angle_course = _compute_sole_angle_course(orientations)
            stride_parameter_dict = {
                **stride_parameter_dict,
                **self._ori_based_parameters(orientations, stride_event_list, angle_course),
            }
        else:
            angle_course = None

        parameters = pd.DataFrame(stride_parameter_dict, index=stride_event_list.index).sort_index(axis=1)
        return {"parameters": parameters, "sole_angle_course": angle_course}

    def _traj_based_parameters(self, positions, stride_event_list, sampling_rate_hz):
        positions = set_correct_index(positions, GF_INDEX)[GF_POS]

        param_dict = {}

        if self._should_calculate("stride_length") or self._should_calculate("gait_velocity"):
            stride_length = _calc_stride_length(positions)
        if self._should_calculate("stride_length"):
            param_dict["stride_length"] = stride_length
        if self._should_calculate("gait_velocity"):
            stride_time_start_col, stride_time_end_col = _get_stride_time_cols(self.expected_stride_type)
            if (
                stride_time_start_col not in stride_event_list.columns
                or stride_time_end_col not in stride_event_list.columns
            ):
                warnings.warn(
                    f"Gait velocity could not be calculated as relevant stride time columns ({stride_time_start_col}, "
                    f"{stride_time_end_col}) are not available."
                )
            else:
                param_dict["gait_velocity"] = stride_length / (
                    (stride_event_list[stride_time_end_col] - stride_event_list[stride_time_start_col])
                    / sampling_rate_hz
                )
        if self._should_calculate("arc_length"):
            param_dict["arc_length"] = _calc_arc_length(positions)
        if self._should_calculate("max_sensor_lift"):
            param_dict["max_sensor_lift"] = _calc_max_sensor_lift(positions)
        if self._should_calculate("max_lateral_excursion"):
            param_dict["max_lateral_excursion"] = _calc_max_lateral_excursion(positions)
        return param_dict

    def _ori_based_parameters(self, orientations, stride_event_list, angle_course):
        param_dict = {}

        if self._should_calculate("ic_angle") and "ic" in stride_event_list.columns:
            ic_relative = (stride_event_list["ic"] - stride_event_list["start"]).astype("Int64")
            param_dict["ic_angle"] = _get_angle_at_index(angle_course, ic_relative)
        else:
            warnings.warn("IC angle could not be calculated as IC event is not available.")

        if self._should_calculate("tc_angle") and "tc" in stride_event_list.columns:
            tc_relative = (stride_event_list["tc"] - stride_event_list["start"]).astype("Int64")
            param_dict["tc_angle"] = _get_angle_at_index(angle_course, tc_relative)
        else:
            warnings.warn("TC angle could not be calculated as TC event is not available.")

        if self._should_calculate("max_orientation_change"):
            param_dict["max_orientation_change"] = _get_max_angle_change(sagittal_angles=angle_course + 90)

        if self._should_calculate("turning_angle"):
            param_dict["turning_angle"] = _calc_turning_angle(orientations)

        return param_dict

    def _should_calculate(self, parameter_name: ParamterNames) -> bool:
        if self.calculate_only is None:
            return True
        return parameter_name in self.calculate_only


def _calc_stride_length(positions: pd.DataFrame) -> pd.Series:
    start = positions.groupby(level="s_id").first()
    end = positions.groupby(level="s_id").last()
    stride_length = end - start
    stride_length = pd.Series(norm(stride_length[["pos_x", "pos_y"]], axis=1), index=stride_length.index)
    return stride_length


def _get_angle_at_index(angle_course: pd.Series, index_per_stride: pd.Series) -> pd.Series:
    indexer = pd.MultiIndex.from_frame(index_per_stride.reset_index())
    return angle_course[indexer].reset_index(level=1, drop=True)


def _get_max_angle_change(sagittal_angles: pd.Series) -> pd.Series:
    # get the maximum change in the orientation in sagittal plane. the sagittal angle is equal to the angle_course +90
    # degrees. This parameter can have a value between 0 and 180.
    max_angle = sagittal_angles.groupby(level="s_id").max()
    min_angle = sagittal_angles.groupby(level="s_id").min()
    return max_angle - min_angle


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
    # Note: We discovered in #187 that due to a series of bugs the sign of these angles is flipped.
    # To follow the convention that the tc_angle should be smaller than 0 (during healthy walking), we multiply here.
    floor_angle *= -1
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
    if positions.empty:
        return pd.Series()
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
