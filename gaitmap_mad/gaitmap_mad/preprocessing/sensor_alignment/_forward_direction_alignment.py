"""Correct for 180 degree misalignments between sensor and foot coordinate frame based on forward direction."""

from typing import Dict, Union

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from tpcp import cf
from typing_extensions import Self

from gaitmap.base import BaseOrientationMethod, BasePositionMethod, BaseSensorAlignment, BaseZuptDetector
from gaitmap.trajectory_reconstruction import MadgwickAHRS, PieceWiseLinearDedriftedIntegration
from gaitmap.utils._algo_helper import invert_result_dictionary, set_params_from_dict
from gaitmap.utils._types import _Hashable
from gaitmap.utils.consts import GRAV_VEC, SF_ACC, SF_COLS, SF_GYR
from gaitmap.utils.datatype_helper import SensorData, SingleSensorData, get_multi_sensor_names, is_sensor_data
from gaitmap.utils.rotations import get_gravity_rotation, rotate_dataset
from gaitmap.zupt_detection import NormZuptDetector


class ForwardDirectionSignAlignment(BaseSensorAlignment):
    """Flip sensor target axis by 0deg or 180deg to align with the sensor frame with the expected forward direction.

    This method applies a fixed 0deg or 180deg flip of the coordinate system to match the sensor frame with the expected
    forward direction. This step is necessary as a subsequent step of other alignment methods which are sign invariant
    like for example an alignment via PCA. Such methods can only align the sensor frame with a given target plane but
    cannot ensure the correct "direction" of the coordinate frame. Therefore, an additional 180deg rotation might be
    necessary. The forward direction is estimated by a strapdown integration of the given IMU data and evaluation of the
    sign of the primary velocity component of the expected forward direction. To ensure only the forward component
    within the sensor frame is considered rotations around the yaw axis within the global frame are ignored. The sign
    (aka 0deg or 180deg flip) is finally estimated by the sign of the mean velocity within the expected forward
    direction.

    Parameters
    ----------
    forward_direction
        axis which corresponds to the expected forward direction e.g. "x"
    rotation_axis
        axis around which the sensor will be flipped, which must be perpendicular to the forward_direction e.g. "z"
    baseline_velocity_threshold
        threshold in m/s^2 around which valocity values are ignored for mean calculation. This is necessary to avoid
        influence of long near zero velocity periods which contain no information about the actual forward direction
    ori_method
        An instance of any available orientation method with the desired parameters set.
        This method is called on the input data to actually calculate the orientation and compensate for gravity.
        Note, the the `initial_orientation` parameter of this method will be overwritten, as this class estimates the
        initial orientation automatically on its own.
    zupt_detector_orientation_init
        An instance of a valid Zupt detector that will be used to estimate the initial orientation for the ori_method.
    pos_method
        An instance of any available position method with the desired parameters set.
        This method is called after the orientation estimation to actually calculate the velocity (and position).
        The provided data is already transformed into the global frame using the orientations calculated by the
        `ori_method`.

    Attributes
    ----------
    aligned_data_
        The rotated sensor data after alignment
    rotation_
        The :class:`~scipy.spatial.transform.Rotation` object tranforming the original data to the aligned data. For
        this class the rotation corresponds either to a 0deg or 180deg rotation.
    is_flipped_
        Boolean indicating if the data is flipped (aka 180 deg rotation required)
    ori_method_
        Reference to the orientation method object after alignment.
        This contains the calculated orientations.
        This is mostly useful for debugging.
    pos_method_
        Reference to the position method object after alignment
        This contains the calculated velocities and positions.
        Note, that these will equivalent to the full sensor trajectory in the world frame.
        This is mostly useful for debugging.

    Other Parameters
    ----------------
    data
        The data passed to the `align` method.
    sampling_rate_hz
        The sampling rate of this data

    Examples
    --------
    Estimate the sign of the forward direction velocity and apply a 0deg or 180deg flip on the data accordingly. This
    step is usually a subsequent step of a previous sign invariant alignment step like the PCA alignment.

    >>> fdsa = ForwardDirectionSignAlignment(forward_direction="x", rotation_axis="z", baseline_velocity_threshold=0.2)
    >>> fdsa = fdsa.align(data, 204.8)
    >>> fdsa.aligned_data_['left_sensor']
    <copy of dataset with axis aligned to the medio-lateral plane>
    ...
    >>> fdsa.is_flipped_  # True when the data was rotated by 180 deg, False afterwise
    True

    Notes
    -----
    This method is usually used as a subsequent step after a sign invariant alignment step like PCA alignment and will
    only apply either a 0deg (aka no rotation) or a 180deg flip on the input data.


    See Also
    --------
    gaitmap.preprocessing.sensor_alignment.PcaAlignment: Details on the PCA based alignment method.

    """

    ori_method: BaseOrientationMethod
    zupt_detector_orientation_init: BaseZuptDetector
    pos_method: BasePositionMethod

    data: SensorData
    sampling_rate_hz: float

    rotation_: Union[Rotation, Dict[_Hashable, Rotation]]
    is_flipped_: Union[bool, Dict[_Hashable, bool]]
    pos_method_: Union[BasePositionMethod, Dict[_Hashable, BasePositionMethod]]
    ori_method_: Union[BaseOrientationMethod, Dict[_Hashable, BaseOrientationMethod]]

    def __init__(
        self,
        forward_direction: str = "x",
        rotation_axis: str = "z",
        baseline_velocity_threshold: float = 0.2,
        ori_method: BaseOrientationMethod = cf(MadgwickAHRS(beta=0.1)),
        zupt_detector_orientation_init: BaseZuptDetector = cf(
            NormZuptDetector(sensor="acc", window_length_s=0.15, inactive_signal_threshold=0.01, metric="variance")
        ),
        pos_method: BasePositionMethod = cf(
            PieceWiseLinearDedriftedIntegration(
                NormZuptDetector(sensor="gyr", window_length_s=0.15, inactive_signal_threshold=15.0, metric="mean"),
                level_assumption=False,
                gravity=GRAV_VEC,
            )
        ),
    ):
        self.forward_direction = forward_direction
        self.rotation_axis = rotation_axis
        self.baseline_velocity_threshold = baseline_velocity_threshold
        self.ori_method = ori_method
        self.zupt_detector_orientation_init = zupt_detector_orientation_init
        self.pos_method = pos_method

    # pylint: disable=arguments-differ
    def align(self, data: SensorData, *, sampling_rate_hz: float, **_) -> Self:
        """Align sensor data."""
        if self.ori_method and not isinstance(self.ori_method, BaseOrientationMethod):
            raise ValueError("The provided `ori_method` must be a child class of `BaseOrientationMethod`.")
        if self.pos_method and not isinstance(self.pos_method, BasePositionMethod):
            raise ValueError("The provided `pos_method` must be a child class of `BasePositionMethod`.")
        if self.rotation_axis.lower() not in "xyz":
            raise ValueError("Invalid rotation axis! Axis must be one of x,y or z!")
        if self.forward_direction.lower() not in "xyz":
            raise ValueError("Invalid forward direction axis! Axis must be one of x,y or z!")
        if self.rotation_axis.lower() == self.forward_direction.lower():
            raise ValueError(
                "Invalid combination of rotation and forward direction axis! They can not be the same! "
                f"Both are `{self.rotation_axis.lower()}`."
            )

        dataset_type = is_sensor_data(data, check_gyr=True, check_acc=True, frame="sensor")
        if dataset_type in ("single", "array"):
            results = self._align_heading_single_sensor(data, sampling_rate_hz)
        else:
            # Multisensor
            result_dict = {
                sensor: self._align_heading_single_sensor(data[sensor], sampling_rate_hz)
                for sensor in get_multi_sensor_names(data)
            }
            results = invert_result_dictionary(result_dict)

        set_params_from_dict(self, results, result_formatting=True)

        return self

    def _align_heading_single_sensor(self, data: SingleSensorData, sampling_rate_hz: float):
        """Align single sensor data."""
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        r = Rotation.from_euler(self.rotation_axis.lower(), 0, degrees=True)
        results_dict = self._forward_direction_is_flipped(data, sampling_rate_hz)
        is_flipped = results_dict["is_flipped"]
        if is_flipped:
            # flip data by 180deg around specified rotation-axis
            r = Rotation.from_euler(self.rotation_axis.lower(), 180, degrees=True)

        return {
            "aligned_data": rotate_dataset(data, r),
            "rotation": r,
            "is_flipped": is_flipped,
            "ori_method": results_dict["ori_method"],
            "pos_method": results_dict["pos_method"],
        }

    def _forward_direction_is_flipped(self, data: SingleSensorData, sampling_rate_hz: float):
        """Estimate if data is 180deg flipped by the sign of the reconstructed forward velocity."""
        # estimate initial orientation for ori method from first static acc region
        zupts_initial_ori = (
            self.zupt_detector_orientation_init.clone().detect(data, sampling_rate_hz=sampling_rate_hz).zupts_
        )
        # only consider the very first static moment
        start, end = zupts_initial_ori.to_numpy()[0]
        first_static_acc_vec = data[SF_ACC].iloc[start:end].median().to_numpy()

        # apply ori-method filter to rotate sensor frame into the global world frame
        # Note we only track the trajectory after the first static moment.
        # This should be sufficient to find the forward direction.
        # TODO: Maybe check if there is sufficient data after the first static moment?
        data_after_first_static = data.iloc[start:]
        initial_orientation = get_gravity_rotation(first_static_acc_vec, GRAV_VEC)
        ori_method = self.ori_method.clone().set_params(initial_orientation=initial_orientation)
        ori_method = ori_method.estimate(data_after_first_static, sampling_rate_hz=sampling_rate_hz)
        acc_data = ori_method.orientation_object_[:-1].apply(data_after_first_static[SF_ACC])
        gyr_data = ori_method.orientation_object_[:-1].apply(data_after_first_static[SF_GYR])

        data_wf = pd.DataFrame(np.column_stack([acc_data, gyr_data]), columns=SF_COLS)

        # apply pos-method to estimate the forward movement direction
        pos_method = self.pos_method.clone().estimate(data_wf, sampling_rate_hz=sampling_rate_hz)

        # to ignore any rotation component around the rotation-/ heading-axis we again apply the inverse of the original
        # rotation around the specified rotation-axis on the estimated world frame velocity
        # We do that by transforming th rotation into euler angels.
        # By defining the order of the angles, we ensure that the rotation we want to rotate around has the range
        # -180 to +180 deg
        rotation_order = {
            "x": "xyz",
            "y": "yzx",
            "z": "zxy",
        }
        # apply inverse of rotations around rotation_axis to ignore this component from the ori_method
        forward_vel_fix_heading = pd.DataFrame(
            Rotation.from_euler(
                self.rotation_axis.lower(),
                ori_method.orientation_object_.as_euler(rotation_order[self.rotation_axis])[:, 0],
            )
            .inv()
            .apply(pos_method.velocity_),
            columns=pos_method.velocity_.columns,
        )["vel_" + self.forward_direction.lower()]

        # Cut out the regions that are for sure not static
        forward_vel_without_baseline = forward_vel_fix_heading[
            forward_vel_fix_heading.abs() >= self.baseline_velocity_threshold
        ]

        return {
            "is_flipped": bool(forward_vel_without_baseline.mean() < 0),
            "ori_method": ori_method,
            "pos_method": pos_method,
        }
