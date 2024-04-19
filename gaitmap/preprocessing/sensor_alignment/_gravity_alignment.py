"""Helpers to rotate the sensor in the predefined gaitmap sensor frame."""

from typing import Union

import numpy as np
import pandas as pd

from gaitmap.utils import rotations
from gaitmap.utils._types import _Hashable
from gaitmap.utils.consts import GRAV_VEC, SF_ACC, SF_GYR
from gaitmap.utils.datatype_helper import SensorData, get_multi_sensor_names, is_sensor_data
from gaitmap.utils.static_moment_detection import METRIC_FUNCTION_NAMES, find_static_samples


def align_dataset_to_gravity(
    dataset: SensorData,
    sampling_rate_hz: float,
    window_length_s: float = 0.7,
    static_signal_th: float = 2.5,
    metric: METRIC_FUNCTION_NAMES = "median",
    gravity: np.ndarray = GRAV_VEC,
) -> SensorData:
    """Align dataset, so that each sensor z-axis (if multiple present in dataset) will be parallel to gravity.

    Median accelerometer vector will be extracted form static windows which will be classified by a sliding window
    with (window_length -1) overlap and a thresholding of the gyro signal norm. This will be performed for each sensor
    in the dataset individually.

    Parameters
    ----------
    dataset : single-sensor dataframe, or multi-sensor dataset
        A single sensor dataset should be represented as a dataframe.
        Multi-sensor datasets should be represented as a dictionary of dataframes, where the keys are the sensor names
        or a pandas dataframe with a multi-index where the first level is the sensor name.

    sampling_rate_hz: float
        Samplingrate of input signal in units of hertz.

    window_length_s : float
        Length of desired window in units of seconds.

    static_signal_th : float
       Threshold to decide whether a window should be considered as static or active. Window will be classified on
       <= threshold on gyro norm

    metric
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
    dataset_type = is_sensor_data(dataset)

    window_length = int(round(window_length_s * sampling_rate_hz))
    acc_vector: Union[np.ndarray, dict[_Hashable, np.ndarray]]
    if dataset_type == "single":
        # get static acc vector
        acc_vector = _get_static_acc_vector(dataset, window_length, static_signal_th, metric)
        # get rotation to gravity
        rotation = rotations.get_gravity_rotation(acc_vector, gravity)
    else:
        # build dict with static acc vectors for each sensor in dataset
        acc_vector = {
            name: _get_static_acc_vector(dataset[name], window_length, static_signal_th, metric)
            for name in get_multi_sensor_names(dataset)
        }
        # build rotation dict for each dataset from acc dict and gravity
        rotation = {
            name: rotations.get_gravity_rotation(acc_vector[name], gravity) for name in get_multi_sensor_names(dataset)
        }

    return rotations.rotate_dataset(dataset, rotation)


def _get_static_acc_vector(
    data: pd.DataFrame, window_length: int, static_signal_th: float, metric: METRIC_FUNCTION_NAMES = "median"
) -> np.ndarray:
    """Extract the mean accelerometer vector describing the static position of the sensor."""
    # find static windows within the gyro data
    static_bool_array, *_ = find_static_samples(data[SF_GYR].to_numpy(), window_length, static_signal_th, metric)

    # raise exception if no static windows could be found with given user settings
    if not any(static_bool_array):
        raise ValueError(
            "No static windows could be found to extract sensor offset orientation. Please check your input data or try"
            " to adapt parameters like window_length, static_signal_th or used metric."
        )

    # get mean acc vector indicating the sensor offset orientation from gravity from static sequences
    return np.median(data[SF_ACC].to_numpy()[static_bool_array], axis=0)
