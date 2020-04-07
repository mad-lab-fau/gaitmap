"""A couple of helper functions that easy the use of the typical gaitmap data formats."""
from typing import Union, Dict, List, Sequence

import numpy as np
import pandas as pd
from typing_extensions import Literal

from gaitmap.utils.consts import SF_ACC, SF_GYR, BF_GYR, BF_ACC

SingleSensorDataset = pd.DataFrame
MultiSensorDataset = Union[pd.DataFrame, Dict[str, SingleSensorDataset]]
Dataset = Union[SingleSensorDataset, MultiSensorDataset]

SingleSensorStridelist = pd.DataFrame
MultiSensorStridelist = Dict[str, pd.DataFrame]
StrideList = Union[SingleSensorDataset, MultiSensorStridelist]


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
    dataset: SingleSensorDataset,
    check_acc: bool = True,
    check_gyr: bool = True,
    frame: Literal["any", "body", "sensor"] = "any",
) -> bool:
    """Check if an object is a valid dataset following all conventions.

    A valid single sensor dataset is:

    - a :class:`pandas.DataFrame`
    - has only a single level of column indices that correspond to the sensor (or feature) axis that are available.

    A valid single sensor dataset in the body frame additionally:

    - contains all columns listed in :obj:`SF_COLS <gaitmap.utils.consts.SF_COLS>`

    A valid single sensor dataset in the sensor frame additionally:

    - contains all columns listed in :obj:`BF_COLS <gaitmap.utils.consts.BF_COLS>`

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
        If you just want to check the datatype and shape, but not for specific column values, set both `check_acc` and
        `check_gyro` to `False`.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_multi_sensor_dataset: Explanation and checks for multi sensor datasets

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


def is_multi_sensor_dataset(
    dataset: MultiSensorDataset,
    check_acc: bool = True,
    check_gyr: bool = True,
    frame: Literal["any", "body", "sensor"] = "any",
) -> bool:
    """Check if an object is a valid multi-sensor dataset.

    A valid multi sensor dataset is:

    - is either a :class:`pandas.DataFrame` with 2 level multi-index as columns or a dictionary of single sensor
      datasets (see :func:`is_single_sensor_dataset <gaitmap.utils.dataset_helper.is_single_sensor_dataset>`)

    In case the dataset is a :class:`pandas.DataFrame` with two levels, the first level is expected to be the names
    of the used sensors.
    In both cases (dataframe or dict), `dataset[<sensor_name>]` is expected to return a valid single sensor
    dataset.
    On each of the these single-sensor datasets,
    :func:`is_single_sensor_dataset <gaitmap.utils.dataset_helper.is_single_sensor_dataset>` is used with the same
    parameters that are used to call this function.

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
        If you just want to check the datatype and shape, but not for specific column values, set both `check_acc` and
        `check_gyro` to `False`.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_single_sensor_dataset: Explanation and checks for single sensor datasets

    """
    if not isinstance(dataset, (pd.DataFrame, dict)):
        return False

    if isinstance(dataset, pd.DataFrame) and (
        (not isinstance(dataset.columns, pd.MultiIndex)) or (dataset.columns.nlevels != 2)
    ):
        # Check that it has multilevel columns
        return False

    keys = get_multi_sensor_dataset_names(dataset)

    if len(keys) == 0:
        return False

    for k in keys:
        if not is_single_sensor_dataset(dataset[k], check_acc=check_acc, check_gyr=check_gyr, frame=frame):
            return False
    return True


def is_single_sensor_stride_list(
    stride_list: SingleSensorStridelist, stride_type: Literal["any", "min_vel"] = "any"
) -> bool:
    if not isinstance(stride_list, pd.DataFrame):
        return False

    columns = stride_list.columns

    if isinstance(columns, pd.MultiIndex):
        return False

    # Check minimal columns exist
    minimal_columns = ["s_id", "start", "end", "gsd_id"]
    if not all(v in columns for v in minimal_columns):
        return False

    if stride_type == "any":
        return True

    # Depending of the stridetype check additional conditions
    additional_columns = {"min_vel": ["pre_ic", "ic", "min_vel", "tc"]}
    start_event = {"min_vel": "min_vel"}

    if stride_type not in additional_columns:
        raise ValueError('The argument `stride_type` must be one of ["any", "min_vel"]')

    if not all(v in columns for v in additional_columns[stride_type]):
        return False

    # Check that the start time corresponds to the correct event
    if len(stride_list) == 0:
        return True
    if not np.array_equal(stride_list["start"].to_numpy(), stride_list[start_event[stride_type]].to_numpy()):
        return False
    return True


def is_multi_sensor_stride_list(
    stride_list: MultiSensorStridelist, stride_type: Literal["any", "segmented", "min_vel", "ic"] = "any"
) -> bool:

    if not isinstance(stride_list, dict):
        return False

    keys = stride_list.keys()

    if len(keys) == 0:
        return False

    for k in keys:
        if not is_single_sensor_stride_list(stride_list[k], stride_type=stride_type):
            return False
    return True


def get_multi_sensor_dataset_names(dataset: MultiSensorDataset) -> Sequence[str]:
    """Get the list of sensor names from a multi-sensor dataset.

    .. warning:
        This will not check, if the input is actually a multi-sensor dataset

    Notes
    -----
    The keys are not guaranteed to be ordered.

    """
    if isinstance(dataset, pd.DataFrame):
        keys = list(set(dataset.columns.get_level_values(0)))
    else:
        # In case it is a dict
        keys = dataset.keys()

    return keys
