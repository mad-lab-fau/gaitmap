"""Helpers to rotate the sensor in the predefined gaitmap sensor frame."""
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from gaitmap.utils import rotations
from gaitmap.utils.array_handling import sliding_window_view
from gaitmap.utils.consts import SF_GYR, SF_ACC, GRAV_VEC
from gaitmap.utils.dataset_helper import (
    get_multi_sensor_dataset_names,
    Dataset,
    is_dataset,
)
from gaitmap.utils.rotations import rotation_from_angle, find_signed_3d_angle
from gaitmap.utils.static_moment_detection import find_static_samples, METRIC_FUNCTION_NAMES
from gaitmap.utils.vector_math import normalize


def align_dataset_to_gravity(
    dataset: Dataset,
    sampling_rate_hz: float,
    window_length_s: float = 0.7,
    static_signal_th: float = 2.5,
    metric: str = "median",
    gravity: np.ndarray = GRAV_VEC,
) -> Dataset:
    """Align dataset, so that each sensor z-axis (if multiple present in dataset) will be parallel to gravity.

    Median accelerometer vector will be extracted form static windows which will be classified by a sliding window
    with (window_length -1) overlap and a thresholding of the gyro signal norm. This will be performed for each sensor
    in the dataset individually.

    Parameters
    ----------
    dataset : gaitmap.utils.dataset_helper.Dataset
        dataframe representing a single or multiple sensors.
        In case of multiple sensors a df with MultiIndex columns is expected where the first level is the sensor name
        and the second level the axis names (all sensor frame axis must be present)

    sampling_rate_hz: float
        Samplingrate of input signal in units of hertz.

    window_length_s : float
        Length of desired window in units of seconds.

    static_signal_th : float
       Threshold to decide whether a window should be considered as static or active. Window will be classified on
       <= threshold on gyro norm

    metric : str, optional
        Metric which will be calculated per window, one of the following strings

        'maximum' (default)
            Calculates maximum value per window
        'mean'
            Calculates mean value per window
        'median'
            Calculates median value per window
        'variance'
            Calculates variance value per window

    gravity : vector with shape (3,), axis ([x, y ,z]), optional
        Expected direction of gravity during rest after the rotation.
        For example if this is `[0, 0, 1]` the sensor will measure +g on the z-axis after rotation (z-axis pointing
        upwards)

    Returns
    -------
    aligned dataset
        This will always be a copy. The original dataframe will not be modified.

    Examples
    --------
    >>> # pd.DataFrame containing one or multiple sensor data streams, each of containing all 6 IMU
    ... # axis (acc_x, ..., gyr_z)
    >>> dataset_df = ...
    >>> align_dataset_to_gravity(dataset_df, window_length_s = 0.7, static_signal_th = 2.0, metric = 'median',
    ... gravity = np.array([0.0, 0.0, 1.0])
    <copy of dataset with all axis aligned to gravity>

    See Also
    --------
    gaitmap.utils.static_moment_detection.find_static_sequences: Details on the used static moment detection function
        for this method.

    """
    dataset_type = is_dataset(dataset)

    window_length = int(round(window_length_s * sampling_rate_hz))

    if dataset_type == "single":
        # get static acc vector
        acc_vector = _get_static_acc_vector(dataset, window_length, static_signal_th, metric)
        # get rotation to gravity
        rotation = rotations.get_gravity_rotation(acc_vector, gravity)
    else:
        # build dict with static acc vectors for each sensor in dataset
        acc_vector = {
            name: _get_static_acc_vector(dataset[name], window_length, static_signal_th, metric)
            for name in get_multi_sensor_dataset_names(dataset)
        }
        # build rotation dict for each dataset from acc dict and gravity
        rotation = {
            name: rotations.get_gravity_rotation(acc_vector[name], gravity)
            for name in get_multi_sensor_dataset_names(dataset)
        }

    return rotations.rotate_dataset(dataset, rotation)


def _get_static_acc_vector(
    data: pd.DataFrame, window_length: int, static_signal_th: float, metric: METRIC_FUNCTION_NAMES = "median"
) -> np.ndarray:
    """Extract the mean accelerometer vector describing the static position of the sensor."""
    # find static windows within the gyro data
    static_bool_array = find_static_samples(data[SF_GYR].to_numpy(), window_length, static_signal_th, metric)

    # raise exception if no static windows could be found with given user settings
    if not any(static_bool_array):
        raise ValueError(
            "No static windows could be found to extract sensor offset orientation. Please check your input data or try"
            " to adapt parameters like window_length, static_signal_th or used metric."
        )

    # get mean acc vector indicating the sensor offset orientation from gravity from static sequences
    return np.median(data[SF_ACC].to_numpy()[static_bool_array], axis=0)


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

    angle_diff = find_signed_3d_angle(gyro_signal_ref[:, :2], gyro_signal_sensor[:, :2], gravity)

    mags = np.max(np.stack([reference_magnitude, sensor_magnitude]), axis=0)

    angle_diff = angle_diff.T[(mags > movement_threshold)].T
    if smoothing_window_size is not None:
        angle_diff = sliding_window_view(
            angle_diff, smoothing_window_size, smoothing_window_size - 1, nan_padding=False
        )
    angle_diff = np.unwrap(angle_diff)
    angle = np.nanmedian(angle_diff)

    return rotation_from_angle(gravity, angle)
