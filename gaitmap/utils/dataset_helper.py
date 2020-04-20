"""A couple of helper functions that easy the use of the typical gaitmap data formats."""
from typing import Union, Dict, List, Sequence, Iterable

import numpy as np
import pandas as pd
from typing_extensions import Literal

from gaitmap.utils.consts import SF_ACC, SF_GYR, BF_GYR, BF_ACC

SingleSensorDataset = pd.DataFrame
MultiSensorDataset = Union[pd.DataFrame, Dict[str, SingleSensorDataset]]
Dataset = Union[SingleSensorDataset, MultiSensorDataset]

SingleSensorStrideList = pd.DataFrame
MultiSensorStrideList = Dict[str, pd.DataFrame]
StrideList = Union[SingleSensorDataset, MultiSensorStrideList]

SingleSensorPositionList = pd.DataFrame
MultiSensorPositionList = Dict[str, pd.DataFrame]
PositionList = Union[SingleSensorPositionList, MultiSensorPositionList]

SingleSensorOrientationList = pd.DataFrame
MultiSensorOrientationList = Dict[str, pd.DataFrame]
OrientationList = Union[SingleSensorOrientationList, MultiSensorOrientationList]


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
    stride_list: SingleSensorStrideList, stride_type: Literal["any", "min_vel"] = "any"
) -> bool:
    """Check if an input is a single-sensor stride list.

    A valid stride list:

    - is a pandas Dataframe with at least the following columns: `["s_id", "start", "end"]`
    - has only a single level column index

    Depending on the type of stride list, further requirements need to be fulfilled:

    min_vel
        A min-vel stride list describes a stride list that defines a stride from one midstance (`min_vel`) to the next.
        This type of stride list can be performed for ZUPT based trajectory estimation.
        It is expected to additionally have the following columns describing relevant stride events: `["pre_ic", "ic",
        "min_vel", "tc"]`.
        See :mod:`~gaitmap.event_detection` for details.
        For this type of stride list it is further tested, that the "start" column is actual identical to the "min_vel"
        column.

    Parameters
    ----------
    stride_list
        The object that should be tested
    stride_type
        The expected stride type of this object.
        If this is "any" only the generally required columns are checked.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_multi_sensor_stride_list: Check for multi-sensor stride lists

    """
    if not isinstance(stride_list, pd.DataFrame):
        return False

    columns = stride_list.columns

    if isinstance(columns, pd.MultiIndex):
        return False

    # Depending of the stridetype check additional conditions
    additional_columns = {"min_vel": ["pre_ic", "ic", "min_vel", "tc"]}
    start_event = {"min_vel": "min_vel"}

    # Check columns exist
    if stride_type != "any" and stride_type not in additional_columns:
        raise ValueError('The argument `stride_type` must be one of ["any", "min_vel"]')
    minimal_columns = ["s_id", "start", "end"]
    all_columns = [*minimal_columns, *additional_columns.get(stride_type, [])]
    if not all(v in columns for v in all_columns):
        return False

    # Check that the start time corresponds to the correct event
    if (
        start_event.get(stride_type, False)
        and len(stride_list) > 0
        and not np.array_equal(stride_list["start"].to_numpy(), stride_list[start_event[stride_type]].to_numpy())
    ):
        return False
    return True


def is_multi_sensor_stride_list(
    stride_list: MultiSensorStrideList, stride_type: Literal["any", "segmented", "min_vel", "ic"] = "any"
) -> bool:
    """Check if an input is a multi-sensor stride list.

    A valid multi-sensor stride list is dictionary of single-sensor stride lists.

    This function :func:`~gaitmap.utils.dataset_helper.is_single_sensor_stride_list` for each of the contained stride
    lists.

    Parameters
    ----------
    stride_list
        The object that should be tested
    stride_type
        The expected stride type of this object.
        If this is "any" only the generally required columns are checked.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_single_sensor_stride_list: Check for multi-sensor stride lists

    """
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
        keys = dataset.columns.unique(level=0)
    else:
        # In case it is a dict
        keys = dataset.keys()

    return keys


def is_single_sensor_position_list(position_list: SingleSensorPositionList) -> bool:
    """Check if an input is a single-sensor position list.

    A valid position list:

    - is a pandas DataFrame with at least the following columns: `["s_id", "sample", "pos_x", "pos_y", "pos_z"]`
    - or a pandas DataFrame with a 2-level MultiIndex with the names `["s_id", "sample"]` and at least to columns
      `["pos_x", "pos_y", "pos_z"]`

    Parameters
    ----------
    position_list
        The object that should be tested

    See Also
    --------
    gaitmap.utils.dataset_helper.is_multi_sensor_position_list: Check for multi-sensor position lists

    """
    if not isinstance(position_list, pd.DataFrame):
        return False

    try:
        position_list = set_correct_index(position_list, ["s_id", "sample"])
    except KeyError:
        return False
    columns = position_list.columns
    expected_columns = ["pos_x", "pos_y", "pos_z"]
    if not all(v in columns for v in expected_columns):
        return False
    return True


def is_multi_sensor_position_list(position_list: MultiSensorPositionList) -> bool:
    """Check if an input is a multi-sensor position list.

    A valid multi-sensor stride list is dictionary of single-sensor position lists.

    This function :func:`~gaitmap.utils.dataset_helper.is_single_sensor_position_list` for each of the contained stride
    lists.

    Parameters
    ----------
    position_list
        The object that should be tested

    See Also
    --------
    gaitmap.utils.dataset_helper.is_single_sensor_position_list: Check for multi-sensor position lists

    """
    if not isinstance(position_list, dict):
        return False

    keys = position_list.keys()

    if len(keys) == 0:
        return False

    for k in keys:
        if not is_single_sensor_position_list(position_list[k]):
            return False
    return True


def is_single_sensor_orientation_list(orientation_list: SingleSensorOrientationList) -> bool:
    """Check if an input is a single-sensor orientation list.

    A valid orientation list:

    - is a pandas DataFrame with at least the following columns: `["s_id", "sample", "qx", "qy", "qz", "qw"]`
    - or a pandas DataFrame with a 2-level MultiIndex with the names `["s_id", "sample"]` and at least to columns
      `["qx", "qy", "qz", "qw"]`

    Parameters
    ----------
    orientation_list
        The object that should be tested

    See Also
    --------
    gaitmap.utils.dataset_helper.is_multi_sensor_orientation_list: Check for multi-sensor orientation lists

    """
    # TODO: Test
    if not isinstance(orientation_list, pd.DataFrame):
        return False
    try:
        orientation_list = set_correct_index(orientation_list, ["s_id", "sample"])
    except KeyError:
        return False
    columns = orientation_list.columns
    expected_columns = ["qx", "qy", "qz", "qw"]
    if not all(v in columns for v in expected_columns):
        return False
    return True


def is_multi_sensor_orientation_list(orientation_list: MultiSensorOrientationList) -> bool:
    """Check if an input is a multi-sensor orientation list.

    A valid multi-sensor stride list is dictionary of single-sensor Orientation lists.

    Function :func:`~gaitmap.utils.dataset_helper.is_single_sensor_orientation_list` for each of the contained stride
    lists.

    Parameters
    ----------
    orientation_list
        The object that should be tested

    See Also
    --------
    gaitmap.utils.dataset_helper.is_single_sensor_orientation_list: Check for multi-sensor orientation lists

    """
    # TODO: Test
    if not isinstance(orientation_list, dict):
        return False

    keys = orientation_list.keys()

    if len(keys) == 0:
        return False

    for k in keys:
        if not is_single_sensor_orientation_list(orientation_list[k]):
            return False
    return True


def set_correct_index(df: pd.DataFrame, index_cols: Iterable, drop_false_index_cols: bool = True) -> pd.DataFrame:
    """Set the correct columns as index, or leave them if they are already in the index.

    Parameters
    ----------
    df
        The dataframe
    index_cols
        A list of names that correspond to the names of the multiindex level names (in order)
    drop_false_index_cols
        If True columns that are set as index in df, but shouldn't will be deleted.
        If False these columns will just be removed from the index and become regular df columns.

    """
    if list(index_cols) == df.index.names:
        return df
    # Find index cols that should not be there
    wrong_index = [i for i, n in enumerate(df.index.names) if n not in index_cols]
    df_just_right_index = df.reset_index(level=wrong_index, drop=drop_false_index_cols)

    drop = False
    if len(wrong_index) == len(df.index.names):
        # In case all cols got dropped, a new range index was generated, which we need to remove again
        drop = True
    return df_just_right_index.reset_index(drop=drop).set_index(index_cols)
