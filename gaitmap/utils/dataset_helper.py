"""A couple of helper functions that easy the use of the typical gaitmap data formats."""
from typing import Union, Dict, Any, List

import pandas as pd
from typing_extensions import Literal

from gaitmap.utils.consts import SF_ACC, SF_GYR, BF_GYR, BF_ACC

SingleSensorDataset = pd.DataFrame
MultiSensorDataset = Dict[str, pd.DataFrame]
Dataset = Union[SingleSensorDataset, MultiSensorDataset]


def _has_sf_cols(columns: List[str], check_acc: bool = True, check_gyr: bool = True):
    """Check if columns contain all required columns for the sensor frame."""
    if check_acc is True:
        if not all(v in columns for v in SF_ACC):
            return False

    if check_gyr is True:
        if not all(v in columns for v in SF_GYR):
            return False

    return True


def _has_bf_cols(columns: List[str], check_acc: bool = True, check_gyr: bool = True):
    """Check if column contain all required columns for the body frame."""
    if check_acc is True:
        if not all(v in columns for v in BF_ACC):
            return False

    if check_gyr is True:
        if not all(v in columns for v in BF_GYR):
            return False

    return True


def is_single_sensor_dataset(
    dataset: Any, check_acc: bool = True, check_gyr: bool = True, frame: Literal["any", "body", "sensor"] = "any"
) -> bool:
    """Check if an object is a valid dataset following all conventions.

    A valid single sensor dataset is:

    - a pd.DataFrame
    - has only a single level of column indices that correspond to the sensor (or feature) axis that are available.

    A valid single sensor dataset in the body frame additionally:

    - contains all columns listed in :obj:`gaitmap.utils.consts.SF_COLS`

    A valid single sensor dataset in the sensor frame additionally:

    - contains all columns listed in :obj:`gaitmap.utils.consts.SB_COLS`

    Parameters
    ----------
    dataset
        Object that should be checked
    check_acc
        If the existence of the correct acc columns should be checked
    check_gyr
        If the existence of the correct gyr columns should be checked
    frame
        The frame the dataset is expected to be in.
        This changes which columns are checked for.
        In case of "any" a dataset is considered valid if it contains the correct columns for one of the two frames.

    """
    if not isinstance(dataset, pd.DataFrame):
        return False

    columns = dataset.columns

    if isinstance(columns, pd.MultiIndex):
        return False

    if frame == "any":
        is_sf = _has_sf_cols(columns, check_acc=check_acc, check_gyr=check_gyr)
        is_bf = _has_bf_cols(columns, check_acc=check_acc, check_gyr=check_gyr)
        return is_sf or is_bf
    if frame == "body":
        return _has_bf_cols(columns, check_acc=check_acc, check_gyr=check_gyr)
    if frame == "sensor":
        return _has_sf_cols(columns, check_acc=check_acc, check_gyr=check_gyr)
    raise ValueError('The argument `frame` must be one of ["any", "body", "sensor"]')
