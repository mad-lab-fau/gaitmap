"""A set of helper functions for the conversion of accelerometer and gyroscope data from the sensor to the body frame.

Definitions can be found in the :ref:`coordinate_systems` guide.
"""
import warnings
from typing import List, Optional

import pandas as pd

from gaitmap.utils.consts import BF_COLS, FSF_FBF_CONVERSION_LEFT, FSF_FBF_CONVERSION_RIGHT, SF_COLS
from gaitmap.utils.datatype_helper import (
    MultiSensorData,
    SingleSensorData,
    get_multi_sensor_names,
    is_multi_sensor_data,
    is_single_sensor_data,
)


def convert_left_foot_to_fbf(data: SingleSensorData):
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
    gaitmap.utils.coordinate_conversion.convert_right_foot_to_fbf: conversion of right foot SingleSensorData
    gaitmap.utils.coordinate_conversion.convert_to_fbf: convert multiple sensors at the same time

    """
    is_single_sensor_data(data, check_acc=False, frame="sensor", raise_exception=True)

    result = pd.DataFrame(columns=BF_COLS)

    # Loop over all axes and convert each one separately
    for sf_col_name in SF_COLS:
        result[FSF_FBF_CONVERSION_LEFT[sf_col_name][1]] = FSF_FBF_CONVERSION_LEFT[sf_col_name][0] * data[sf_col_name]

    return result


def convert_right_foot_to_fbf(data: SingleSensorData):
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
    gaitmap.utils.coordinate_conversion.convert_left_foot_to_fbf: conversion of left foot SingleSensorData
    gaitmap.utils.coordinate_conversion.convert_to_fbf: convert multiple sensors at the same time

    """
    is_single_sensor_data(data, check_acc=False, frame="sensor", raise_exception=True)

    result = pd.DataFrame(columns=BF_COLS)

    # Loop over all axes and convert each one separately
    for sf_col_name in SF_COLS:
        result[FSF_FBF_CONVERSION_RIGHT[sf_col_name][1]] = FSF_FBF_CONVERSION_RIGHT[sf_col_name][0] * data[sf_col_name]

    return result


def convert_to_fbf(
    data: MultiSensorData,
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

    >>> dataset = ... # Sensordata in FSF
    >>> fbf_dataset = convert_to_fbf(dataset, left_like="left_", right_like="right_")

    Alternatively, you can specify the full sensor names.

    >>> fbf_dataset = convert_to_fbf(dataset, left=["left_sensor"], right_sensor=["right_sensor"])

    See Also
    --------
    gaitmap.utils.coordinate_conversion.convert_left_foot_to_fbf: conversion of left foot SingleSensorData
    gaitmap.utils.coordinate_conversion.convert_right_foot_to_fbf: conversion of right foot SingleSensorData

    """
    if not is_multi_sensor_data(data, frame="sensor"):
        raise ValueError("No valid FSF MultiSensorDataset supplied.")

    if (left and left_like) or (right and right_like) or not any((left, left_like, right, right_like)):
        raise ValueError(
            "You need to either supply a list of names via the `left` or `right` arguments, or a single string for the "
            "`left_like` or `right_like` arguments, but not both!"
        )

    left_foot = _handle_foot(left, left_like, data, rot_func=convert_left_foot_to_fbf)
    right_foot = _handle_foot(right, right_like, data, rot_func=convert_right_foot_to_fbf)

    sensor_names = get_multi_sensor_names(data)
    result = {k: data[k] for k in sensor_names}
    result = {**result, **left_foot, **right_foot}

    # If original data is not synchronized (dictionary), return as dictionary
    if isinstance(data, dict):
        return result
    # For synchronized sensors, return as MultiIndex dataframe
    result_df = pd.concat(result, axis=1)
    # restore original order
    return result_df[sensor_names]


def _handle_foot(foot, foot_like, data, rot_func):
    result = {}
    if foot_like:
        foot = [sensor for sensor in get_multi_sensor_names(data) if foot_like in sensor]
        if not foot:
            warnings.warn(
                "The substring {} is not contained in any sensor name. Available sensor names are: {}".format(
                    foot_like, get_multi_sensor_names(data)
                )
            )
    foot = foot or []
    for s in foot:
        if s not in data:
            raise KeyError("Sensordata contains no sensor with name " + s)
        result[s] = rot_func(data[s])
    return result
