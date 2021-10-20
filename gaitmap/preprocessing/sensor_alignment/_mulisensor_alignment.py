"""A implementation to align multiple sensors attached to a shared rigid body."""

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from gaitmap.utils.array_handling import sliding_window_view
from gaitmap.utils.consts import GRAV_VEC
from gaitmap.utils.rotations import find_signed_3d_angle, rotation_from_angle
from gaitmap.utils.vector_math import normalize


def align_heading_of_sensors(
    gyro_signal_sensor: np.ndarray,
    gyro_signal_ref: np.ndarray,
    movement_threshold: float = 150,
    smoothing_window_size: Optional[int] = None,
) -> Rotation:
    """Align the heading (rotation in the ground plane) of multiple sensors attached to the same rigid body.

    This function can be used to find relative heading of two sensors attached to the same body segment.
    It is assumed, that they are already aligned so that the gravity vector aligns with the z-axis.
    To find the alignment in the ground-plane it is assumed that both sensors measure roughly the same angular velocity.
    Then the optimal rotation around the gravity axis is determined to align both measurements.
    This rotation is the median angle between the gyroscope vectors in the ground-plane.
    As this angle can vary highly for small values due to noise, the `movement_threshold` is used to select only active
    regions of the signal for the comparison.
    In some cases noise and signal artifacts might still effect the final result.
    In these cases the angle smoothing option should be use to remove outliers using a moving median filter on the
    calculated angle, before they are unwraped.
    This functionality can be controlled by the `smoothing_window_size`.
    Note that the effect of smoothing was not investigated in detail and it is advisable to calculate and visualise the
    residual distance between the sensor signals to catch potential misalignments.

    Parameters
    ----------
    gyro_signal_sensor : array with shape (n, 3)
        The gyro signal in deg/s of the sensor that should be aligned
    gyro_signal_ref : array with shape (n, 3)
        The gyro signal in deg/s of the reference sensor
    movement_threshold
        Minimal required gyro value in the xy-plane.
        The unit will depend on the unit of the gyroscope.
        The default value is assumes deg/s as the unit.
        Values below this threshold are not considered in the calculation of the optimal alignment.
    smoothing_window_size
        Size of the moving median filter that is applied to the extracted angles to remove outliers.
        Optimal size should be determined empirical.
        In case it is None, not filter is applied.

    Returns
    -------
    relative_heading
        The required rotation to align the sensor to the reference

    Notes
    -----
    This function could be used to rotate multiple sensors attached to the same foot in a common reference coordinate
    system to compare the raw signal values.
    In a 2 step process, one could first align all signals to gravity and then apply `align_heading_of_sensors` to
    find the missing rotation in the xy-plane.

    """
    gravity = normalize(GRAV_VEC)

    reference_magnitude = np.sqrt(gyro_signal_ref[:, 0] ** 2 + gyro_signal_ref[:, 1] ** 2)
    sensor_magnitude = np.sqrt(gyro_signal_sensor[:, 0] ** 2 + gyro_signal_sensor[:, 1] ** 2)

    angle_diff = np.asarray(find_signed_3d_angle(gyro_signal_ref[:, :2], gyro_signal_sensor[:, :2], gravity))

    mags = np.max(np.stack([reference_magnitude, sensor_magnitude]), axis=0)

    angle_diff = angle_diff.T[(mags > movement_threshold)].T
    if smoothing_window_size is not None:
        angle_diff = sliding_window_view(
            angle_diff, smoothing_window_size, smoothing_window_size - 1, nan_padding=False
        )
    angle_diff = np.unwrap(angle_diff)
    angle = np.nanmedian(angle_diff)

    return rotation_from_angle(gravity, angle)
