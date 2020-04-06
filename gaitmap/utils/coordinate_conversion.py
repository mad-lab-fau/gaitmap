"""A set of helper functions for the conversion of accelerometer
and gyroscope data from the sensor to the body frame
Definitions can be found in http://newgaitpipeline.mad-pages.informatik.uni-erlangen.de/gaitmap/guides/Coordinate-Systems.html"""

import pandas as pd
from gaitmap.utils.consts import SF_COLS, BF_COLS


def convert_left_foot(data: pd.DataFrame):
    """ Converts the axes from the sensor frame to the body frame for the left foot.

    Parameters
    ----------
    data
        raw data

    Returns
    -------
        converted data
    """

    # Definition of the conversion of all axes for the left foot
    conversion_left = {
        "acc_x": (1, "acc_pa"),
        "acc_y": (1, "acc_ml"),
        "acc_z": (-1, "acc_si"),
        "gyr_x": (-1, "gyr_pa"),
        "gyr_y": (-1, "gyr_ml"),
        "gyr_z": (-1, "gyr_si")
    }

    result = pd.DataFrame(columns=BF_COLS)

    # Loop over all axes and convert each one separately
    for sf_col_name in SF_COLS:
        result[conversion_left[sf_col_name][1]] = conversion_left[sf_col_name][0] * data[sf_col_name]

    return result


def convert_right_foot(data: pd.DataFrame):
    """ Converts the axes from the sensor frame to the body frame for the right foot.

    Parameters
    ----------
    data
        raw data

    Returns
    -------
        converted data
    """

    # Definition of the conversion of all axes for the right foot
    conversion_right = {
        "acc_x": (1, "acc_pa"),
        "acc_y": (-1, "acc_ml"),
        "acc_z": (-1, "acc_si"),
        "gyr_x": (1, "gyr_pa"),
        "gyr_y": (-1, "gyr_ml"),
        "gyr_z": (1, "gyr_si")
    }

    result = pd.DataFrame(columns=BF_COLS)

    # Loop over all axes and convert each one separately
    for sf_col_name in SF_COLS:
        result[conversion_right[sf_col_name][1]] = conversion_right[sf_col_name][0] * data[sf_col_name]

    return result
