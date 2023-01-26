"""Internal helpers for dataset validation."""
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import pandas as pd
from typing_extensions import Literal

from gaitmap.utils._types import _Hashable
from gaitmap.utils.consts import BF_ACC, BF_GYR, SF_ACC, SF_GYR
from gaitmap.utils.exceptions import ValidationError

_ALLOWED_FRAMES = ["any", "body", "sensor"]
_ALLOWED_FRAMES_TYPE = Literal["any", "body", "sensor"]  # pylint: disable=invalid-name
_ALLOWED_STRIDE_TYPE = Literal["any", "segmented", "min_vel", "ic"]  # pylint: disable=invalid-name
_ALLOWED_TRAJ_LIST_TYPES = Literal["stride", "roi", "gs", "any_roi"]  # pylint: disable=invalid-name


def _get_expected_dataset_cols(
    frame: Literal["sensor", "body"], check_acc: bool = True, check_gyr: bool = True
) -> List:
    expected_cols = []
    if frame == "sensor":
        acc = SF_ACC
        gyr = SF_GYR
    elif frame == "body":
        acc = BF_ACC
        gyr = BF_GYR
    else:
        raise ValueError('`frame must be one of ["sensor", "body"]')
    if check_acc is True:
        expected_cols.extend(acc)
    if check_gyr is True:
        expected_cols.extend(gyr)
    return expected_cols


def _assert_is_dtype(obj, dtype: Union[type, Tuple[type, ...]]):
    """Check if an object has a specific dtype."""
    if not isinstance(obj, dtype):
        raise ValidationError(f"The dataobject is expected to be one of ({dtype},). But it is a {type(obj)}")


def _assert_has_multindex_cols(df: pd.DataFrame, nlevels: int = 2, expected: bool = True):
    """Check if a pd.DataFrame has a multiindex as columns.

    Parameters
    ----------
    df
        The dataframe to check
    nlevels
        If MultiIndex is expected, how many level should the MultiIndex have
    expected
        If the df is expected to have a MultiIndex or not

    """
    has_multiindex = isinstance(df.columns, pd.MultiIndex)
    if has_multiindex is not expected:
        if expected is False:
            raise ValidationError(
                "The dataframe is expected to have a single level of columns. "
                "But it has a MultiIndex with {} levels.".format(df.columns.nlevels)
            )
        raise ValidationError(
            "The dataframe is expected to have a MultiIndex with {} levels as columns. "
            "It has just a single normal column level.".format(nlevels)
        )
    if has_multiindex is True and not df.columns.nlevels == nlevels:
        raise ValidationError(
            "The dataframe is expected to have a MultiIndex with {} levels as columns. "
            "It has a MultiIndex with {} levels.".format(nlevels, df.columns.nlevels)
        )


def _assert_has_columns(df: pd.DataFrame, columns_sets: Sequence[Union[List[_Hashable], List[str]]]):
    """Check if the dataframe has at least all columns sets.

    Examples
    --------
    >>> df = pd.DataFrame()
    >>> df.columns = ["col1", "col2"]
    >>> _assert_has_columns(df, [["other_col1", "other_col2"], ["col1", "col2"]])
    >>> # This raises no error, as df contains all columns of the second set

    """
    columns_sets_as_set = [set(col_set) for col_set in columns_sets]
    columns = df.columns
    result = False
    for col_set in columns_sets_as_set:
        result = result or all(v in columns for v in col_set)

    if result is False:
        if len(columns_sets) == 1:
            helper_str = f"columns: {columns_sets[0]}"
        else:
            helper_str = f"one of the following sets of columns: {columns_sets}"
        raise ValidationError(
            "The dataframe is expected to have {}. Instead it has the following columns: {}".format(
                helper_str, list(df.columns)
            )
        )


def _assert_has_index_columns(df: pd.DataFrame, index_cols: Iterable[_Hashable]):
    ex_index_cols = list(index_cols)
    ac_index_cols = list(df.index.names)
    if ex_index_cols != ac_index_cols:
        raise ValidationError(
            "The dataframe is expected to have exactly the following index columns ({}), "
            "but it has {}".format(index_cols, df.index.name)
        )


# This function exists to avoid cyclic imports in this module
def _get_multi_sensor_data_names(dataset: Union[dict, pd.DataFrame]) -> Sequence[str]:
    if isinstance(dataset, pd.DataFrame):
        keys = dataset.columns.unique(level=0)
    else:
        # In case it is a dict
        keys = dataset.keys()

    return keys


def _assert_multisensor_is_not_empty(obj: Union[pd.DataFrame, Dict]):
    sensors = _get_multi_sensor_data_names(obj)
    if len(sensors) == 0:
        raise ValidationError("The provided multi-sensor object does not contain any data/contains no sensors.")
