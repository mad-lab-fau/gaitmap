"""A set of helper functions for the conversion of accelerometer and gyroscope data from the sensor to the body frame.

Definitions can be found in
http://newgaitpipeline.mad-pages.informatik.uni-erlangen.de/gaitmap/guides/Coordinate-Systems.html.
"""

from typing import Optional, List

import pandas as pd

from gaitmap.utils.consts import SF_COLS, BF_COLS
from gaitmap.utils.dataset_helper import is_multi_sensor_dataset, SingleSensorDataset, MultiSensorDataset


def convert_left_foot_to_fbf(data: SingleSensorDataset):
    """Convert the axes from the left foot sensor frame to the foot body frame (FBF).

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

    """
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

    """
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


def convert_to_fbf(data: MultiSensorDataset, left: Optional[List[str]] = None, right: Optional[List[str]] = None):
    """Convert the axes from the sensor frame to the body frame for one MultiSensorDataset.

    Parameters
    ----------
    data
        MultiSensorDataset
    left
        List of strings indicating sensor names which will be rotated using definition of left conversion
    right
        List of strings indicating sensor names which will be rotated using definition of right conversion

    Returns
    -------
    converted MultiSensorDataset

    See Also
    --------
    gaitmap.utils.coordinate_conversion.convert_left_foot_to_fbf: conversion of left foot SingleSensorDataset
    gaitmap.utils.coordinate_conversion.convert_right_foot_to_fbf: conversion of right foot SingleSensorDataset

    """
    # TODO: Support also single sensor Dataframe here?
    if not is_multi_sensor_dataset(data):
        raise TypeError("No MultiSensorDataset supplied.")

    if left is None and right is None:
        raise ValueError("Invalid inputs: Neither left nor right sensor names specified.")

    result = dict()

    # Loop through defined sensors
    # Add results to a new dictionary with sensor names as keys
    if left is not None:
        for ls in left:
            result[ls] = convert_left_foot_to_fbf(data[ls])

    if right is not None:
        for rs in right:
            result[rs] = convert_right_foot_to_fbf(data[rs])

    if result:
        # If original data is not synchronized (dictionary), return as dictionary
        if isinstance(data, dict):
            return result
        # For synchronized sensors, return as MultiIndex dataframe
        if result:
            return pd.concat(result, axis=1)

    return None
