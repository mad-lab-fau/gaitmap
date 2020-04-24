"""A set of helper functions for the conversion of accelerometer and gyroscope data from the sensor to the body frame.

Definitions can be found in
http://newgaitpipeline.mad-pages.informatik.uni-erlangen.de/gaitmap/guides/Coordinate-Systems.html.
"""
import warnings
from typing import Optional, List

import pandas as pd

from gaitmap.utils.consts import SF_COLS, BF_COLS
from gaitmap.utils.dataset_helper import (
    is_multi_sensor_dataset,
    SingleSensorDataset,
    MultiSensorDataset,
    get_multi_sensor_dataset_names,
    is_single_sensor_dataset,
)


def convert_left_foot_to_fbf(data: SingleSensorDataset):
    """Convert the axes from the left foot sensor frame to the foot body frame (FBF).

    This function assumes that your dataset is already aligned to the gaitmap FSF.

    Parameters
    ----------
    data
        raw data frame containing acc and gyr data

    Returns
    -------
    converted data frame

    See Also
    --------
    gaitmap.utils.coordinate_conversion.convert_right_foot_to_fbf: conversion of right foot SingleSensorDataset
    gaitmap.utils.coordinate_conversion.convert_to_fbf: convert multiple sensors at the same time

    """
    if not is_single_sensor_dataset(data, frame="sensor"):
        raise ValueError("No valid FSF SingleSensorDataset supplied.")
    # Definition of the conversion of all axes for the left foot
    # TODO: Put into consts.py
    conversion_left = {
        "acc_x": (1, "acc_pa"),
        "acc_y": (1, "acc_ml"),
        "acc_z": (-1, "acc_si"),
        "gyr_x": (-1, "gyr_pa"),
        "gyr_y": (-1, "gyr_ml"),
        "gyr_z": (-1, "gyr_si"),
    }

    result = pd.DataFrame(columns=BF_COLS)

    # Loop over all axes and convert each one separately
    for sf_col_name in SF_COLS:
        result[conversion_left[sf_col_name][1]] = conversion_left[sf_col_name][0] * data[sf_col_name]

    return result


def convert_right_foot_to_fbf(data: SingleSensorDataset):
    """Convert the axes from the right foot sensor frame to the foot body frame (FBF).

    This function assumes that your dataset is already aligned to the gaitmap FSF.

    Parameters
    ----------
    data
        raw data frame containing acc and gyr data

    Returns
    -------
    converted data frame

    See Also
    --------
    gaitmap.utils.coordinate_conversion.convert_left_foot_to_fbf: conversion of left foot SingleSensorDataset
    gaitmap.utils.coordinate_conversion.convert_to_fbf: convert multiple sensors at the same time

    """
    if not is_single_sensor_dataset(data, frame="sensor"):
        raise ValueError("No valid FSF SingleSensorDataset supplied.")
    # Definition of the conversion of all axes for the right foot
    # TODO: Put into consts.py
    conversion_right = {
        "acc_x": (1, "acc_pa"),
        "acc_y": (-1, "acc_ml"),
        "acc_z": (-1, "acc_si"),
        "gyr_x": (1, "gyr_pa"),
        "gyr_y": (-1, "gyr_ml"),
        "gyr_z": (1, "gyr_si"),
    }

    result = pd.DataFrame(columns=BF_COLS)

    # Loop over all axes and convert each one separately
    for sf_col_name in SF_COLS:
        result[conversion_right[sf_col_name][1]] = conversion_right[sf_col_name][0] * data[sf_col_name]

    return result


def convert_to_fbf(
    data: MultiSensorDataset,
    left: Optional[List[str]] = None,
    right: Optional[List[str]] = None,
    right_like: str = None,
    left_like: str = None,
):
    """Convert the axes from the sensor frame to the body frame for one MultiSensorDataset.

    This function assumes that your dataset is already aligned to the gaitmap FSF.
    Sensors that should not be transformed are kept untouched.
    Note, that the column names of all transformed dataset is changed to the respective body frame names.

    This function can handle multiple left and right sensors at the same time.

    Parameters
    ----------
    data
        MultiSensorDataset
    left
        List of strings indicating sensor names which will be rotated using definition of left conversion.
        This option can not be used in combination with `left_like`.
    right
        List of strings indicating sensor names which will be rotated using definition of right conversion
        This option can not be used in combination with `right_like`.
    left_like
        Consider all sensors containing this string in the name as left foot sensors.
        This option can not be used in combination with `left`.
    right_like
        Consider all sensors containing this string in the name as right foot sensors.
        This option can not be used in combination with `right`.


    Returns
    -------
    converted MultiSensorDataset

    Examples
    --------
    These examples assume that your dataset has two sensors called `left_sensor` and `right_sensor`.
    >>> dataset = ... # Dataset in FSF
    >>> fbf_dataset = convert_to_fbf(dataset, left_like="left_", right_like="right_")

    Alternativly, you can specify the full sensor names.
    >>> fbf_dataset = convert_to_fbf(dataset, left=["left_sensor"], right_sensor=["right_sensor"])

    See Also
    --------
    gaitmap.utils.coordinate_conversion.convert_left_foot_to_fbf: conversion of left foot SingleSensorDataset
    gaitmap.utils.coordinate_conversion.convert_right_foot_to_fbf: conversion of right foot SingleSensorDataset

    """
    if not is_multi_sensor_dataset(data, frame="sensor"):
        raise ValueError("No valid FSF MultiSensorDataset supplied.")

    if (left and left_like) or (right and right_like) or not any((left, left_like, right, right_like)):
        raise ValueError(
            "You need to either supply a list of names via the `left` or `right` arguments, or a single string for the "
            "`left_like` or `right_like` arguments, but not both!"
        )

    left_foot = _handle_foot(left, left_like, data, rot_func=convert_left_foot_to_fbf)
    right_foot = _handle_foot(right, right_like, data, rot_func=convert_right_foot_to_fbf)

    sensor_names = get_multi_sensor_dataset_names(data)
    result = {k: data[k] for k in sensor_names}
    result = {**result, **left_foot, **right_foot}

    # If original data is not synchronized (dictionary), return as dictionary
    if isinstance(data, dict):
        return result
    # For synchronized sensors, return as MultiIndex dataframe
    df = pd.concat(result, axis=1)
    # restore original order
    return df[sensor_names]


def _handle_foot(foot, foot_like, data, rot_func):
    result = dict()
    if foot_like:
        foot = [sensor for sensor in get_multi_sensor_dataset_names(data) if foot_like in sensor]
        if not foot:
            warnings.warn(
                "The substring {} is not contained in any sensor name. Available sensor names are: {}".format(
                    foot_like, get_multi_sensor_dataset_names(data)
                )
            )
    foot = foot or []
    for s in foot:
        if s not in data:
            raise KeyError("Dataset contains no sensor with name " + s)
        result[s] = rot_func(data[s])
    return result
