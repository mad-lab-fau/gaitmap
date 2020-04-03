"""A couple of helper functions that easy the use of the typical gaitmap data formats."""
from typing import Union, Dict, Any, List

import pandas as pd
from typing_extensions import Literal

from gaitmap.utils.consts import SF_ACC, SF_GYR, BF_GYR, BF_ACC

SingleSensorDataset = pd.DataFrame
MultiSensorDataset = Dict[str, pd.DataFrame]
Dataset = Union[SingleSensorDataset, MultiSensorDataset]


def _has_sf_cols(columns: List[str], check_acc: bool = True, check_gyr: bool = True):
    if check_acc is True:
        if not all(v in columns for v in SF_ACC):
            return False

    if check_gyr is True:
        if not all(v in columns for v in SF_GYR):
            return False

    return True


def _has_bf_cols(columns: List[str], check_acc: bool = True, check_gyr: bool = True):
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
    """Check if an object is a valid dataset following all conventions."""

    if not isinstance(dataset, pd.DataFrame):
        return False

    columns = dataset.columns

    if isinstance(columns, pd.MultiIndex):
        return False

    if frame == "any":
        return _has_sf_cols(columns, check_acc=check_acc, check_gyr=check_gyr) or _has_bf_cols(
            columns, check_acc=check_acc, check_gyr=check_gyr
        )
    elif frame == "body":
        return _has_bf_cols(columns, check_acc=check_acc, check_gyr=check_gyr)
    elif frame == "sensor":
        return _has_sf_cols(columns, check_acc=check_acc, check_gyr=check_gyr)

    else:
        raise ValueError('The argument `frame` must be one of ["any", "body", "sensor"]')
