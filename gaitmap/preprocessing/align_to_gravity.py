from typing import Optional
import numpy as np
import pandas as pd
from gaitmap.utils.consts import SF_GYR, SF_ACC
from gaitmap.utils.static_moment_detection import find_static_sequences
from gaitmap.utils import rotations


def align_dataset(
    dataset: pd.DataFrame,
    window_length: int,
    static_signal_th: float,
    metric: Optional[str] = "maximum",
    gravity: Optional[np.ndarray] = np.array([0.0, 0.0, 1.0]),
) -> pd.DataFrame:
    """Align dataset, so that each sensor z-axis (if multiple present in dataset) will be parallel to gravity.

    Mean accelerometer vector will be extracted form static windows which will be classified by a sliding window
    with (window_length -1) overlap and a thresholding of the gyro signal norm. This will be performed for each sensor
    in the dataset individually.

    Parameters
    ----------
    dataset : pd.DataFrame
        dataframe representing a single or multiple sensors.
        In case of multiple sensors a df with MultiIndex columns is expected where the first level is the sensor name
        and the second level the axis names (all sensor frame axis must be present)

    window_length : int
        Length of desired window in units of samples.

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

    gravity : np.ndarray, optional
        vector with shape (3,), axis ([x, y ,z])
        Expected measured signal on accelerometer if only gravity would be present.

    Returns
    -------
    aligned dataset
        This will always be a copy. The original dataframe will not be modified.

    Examples
    --------
    >>> dataset_df = ...  # pd.DataFrame containing one or multiple sensor data streams, each of containing all 6 IMU
    ... axis (acc_x, ..., gyr_z)
    >>> align_dataset(sensor_data_df, window_length = 100, static_signal_th = 1.5, metric = 'maximum',
    ... gravity = np.array([0.0, 0.0, 1.0])
    <copy of dataset with all axis aligned to gravity>

    See Also
    --------
    gaitmap.utils.static_moment_detection.find_static_sequences: Details on the used static moment detection function
        for this method.

    """
    multi_index = dataset.columns.nlevels > 1
    if not multi_index:
        return _align_sensor(dataset, window_length, static_signal_th, metric, gravity)

    # build dict with static acc vectors for each sensor in dataset
    acc_vector = {
        name: _get_static_acc_vector(dataset[name], window_length, static_signal_th, metric)
        for name in dataset.columns.levels[0]
    }

    # build rotation dict for each dataset from acc dict and gravity
    rotation = {name: rotations.get_gravity_rotation(acc_vector[name], gravity) for name in dataset.columns.levels[0]}

    return rotations.rotate_dataset(dataset, rotation)


def _get_static_acc_vector(
    data: pd.DataFrame, window_length: int, static_signal_th: float, metric: str = "maximum"
) -> np.ndarray:
    """Extract the mean accelerometer vector describing the static position of the sensor."""
    # find static windows within the gyro data
    static_windows = find_static_sequences(data[SF_GYR].to_numpy(), window_length, static_signal_th, metric)

    # raise exception if no static windows could be found with given user settings
    if static_windows.size == 0:
        raise ValueError(
            "No static windows could be found to extract sensor offset orientation. Please check your input data or try"
            " to adapt parameters like window_length, static_signal_th or used metric."
        )

    # generate indices where data can be considered static
    static_indices = np.concatenate([np.arange(start, end + 1) for start, end in static_windows])

    # get mean acc vector indicating the sensor offset orientation from gravity from static sequences
    return np.mean(data[SF_ACC].to_numpy()[static_indices], axis=0)


def _align_sensor(
    data: pd.DataFrame,
    window_length: int,
    static_signal_th: float,
    metric: Optional[str] = "maximum",
    gravity: Optional[np.ndarray] = np.array([0.0, 0.0, 1.0]),
) -> pd.DataFrame:
    """Align the data of a single sensor dataframe with acc and gyro to gravity."""
    # get mean acc vector indicating the sensor offset orientation from gravity from static sequences
    acc_vector = _get_static_acc_vector(data, window_length, static_signal_th, metric)

    # get rotation to gravity
    r = rotations.get_gravity_rotation(acc_vector, gravity)

    # apply rotation to data
    return rotations.rotate_dataset(data, r)
