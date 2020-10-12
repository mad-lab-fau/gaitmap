"""A couple of helper functions that easy the use of the typical gaitmap data formats."""
from typing import Union, Dict, List, Sequence, Iterable, Hashable, Optional

import numpy as np
import pandas as pd
from typing_extensions import Literal

from gaitmap.utils.consts import (
    SF_ACC,
    SF_GYR,
    BF_GYR,
    BF_ACC,
    SL_COLS,
    SL_ADDITIONAL_COLS,
    GF_POS,
    GF_VEL,
    GF_ORI,
    ROI_ID_COLS,
    SL_INDEX,
)
from gaitmap.utils.exceptions import ValidationError

SingleSensorDataset = pd.DataFrame
MultiSensorDataset = Union[pd.DataFrame, Dict[Hashable, SingleSensorDataset]]
Dataset = Union[SingleSensorDataset, MultiSensorDataset]

SingleSensorStrideList = pd.DataFrame
MultiSensorStrideList = Dict[Hashable, pd.DataFrame]
StrideList = Union[SingleSensorStrideList, MultiSensorStrideList]

SingleSensorRegionsOfInterestList = pd.DataFrame
MultiSensorRegionsOfInterestList = Dict[Hashable, pd.DataFrame]
RegionsOfInterestList = Union[SingleSensorRegionsOfInterestList, MultiSensorRegionsOfInterestList]

SingleSensorPositionList = pd.DataFrame
MultiSensorPositionList = Dict[Hashable, pd.DataFrame]
PositionList = Union[SingleSensorPositionList, MultiSensorPositionList]

SingleSensorVelocityList = pd.DataFrame
MultiSensorVelocityList = Dict[str, pd.DataFrame]
VelocityList = Union[SingleSensorVelocityList, MultiSensorVelocityList]

SingleSensorOrientationList = pd.DataFrame
MultiSensorOrientationList = Dict[Hashable, pd.DataFrame]
OrientationList = Union[SingleSensorOrientationList, MultiSensorOrientationList]

_ALLOWED_FRAMES = ["any", "body", "sensor"]
_ALLOWED_FRAMES_TYPE = Literal["any", "body", "sensor"]


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


def _assert_is_dtype(obj, dtype: Union[type, Iterable[type]]):
    """Check if an object has a specific dtype."""
    if not isinstance(obj, dtype):
        raise ValidationError("The dataobject is expected to be one of ({},). But it is a {}".format(dtype, type(obj)))


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
    if has_multiindex is True:
        if not df.columns.nlevels == nlevels:
            raise ValidationError(
                "The dataframe is expected to have a MultiIndex with {} levels as columns. "
                "It has a MultiIndex with {} levels.".format(nlevels, df.columns.nlevels)
            )


def _assert_has_columns(df: pd.DataFrame, columns_sets: List[List[Hashable]]):
    """Check if the dataframe has at least all columns sets.

    Examples
    --------
    >>> df = pd.DataFrame()
    >>> df.columns = ["col1", "col2"]
    >>> _assert_has_columns(df, [["other_col1", "other_col2"], ["col1", "col2"]])
    >>> # This raises no error, as df contains all columns of the second set

    """
    columns = df.columns
    result = False
    for col_set in columns_sets:
        result = result or all(v in columns for v in col_set)

    if result is False:
        if len(columns_sets) == 1:
            helper_str = "columns: {}".format(columns_sets[0])
        else:
            helper_str = "one of the following sets of columns: {}".format(columns_sets)
        raise ValidationError(
            "The dataframe is expected to have {}. Instead it has the following columns: {}".format(
                helper_str, list(df.columns)
            )
        )


def _assert_has_index_columns(df: pd.DataFrame, index_cols: Iterable[Hashable]):
    ex_index_cols = list(index_cols)
    ac_index_cols = list(df.index.names)
    if ex_index_cols != ac_index_cols:
        raise ValidationError(
            "The dataframe is expected to have exactly the following index columns ({}), "
            "but it has {}".format(index_cols, df.index.name)
        )


def _assert_multisensor_is_not_empty(obj: Union[pd.DataFrame, Dict]):
    sensors = get_multi_sensor_dataset_names(obj)
    if len(sensors) == 0:
        raise ValidationError("The provided multi-sensor object does not contain any data/contains no sensors.")


def is_single_sensor_dataset(
    dataset: SingleSensorDataset,
    check_acc: bool = True,
    check_gyr: bool = True,
    frame: _ALLOWED_FRAMES_TYPE = "any",
    raise_exception: bool = False,
) -> Optional[bool]:
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
    raise_exception
        If True an exception is raised if the object does not pass the validation.
        If False, the function will return simply True or False.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_multi_sensor_dataset: Explanation and checks for multi sensor datasets

    """
    if frame not in _ALLOWED_FRAMES:
        raise ValueError("The argument `frame` must be one of {}".format(_ALLOWED_FRAMES))
    try:
        _assert_is_dtype(dataset, pd.DataFrame)
        _assert_has_multindex_cols(dataset, expected=False)

        if frame == "any":
            _assert_has_columns(
                dataset,
                [
                    _get_expected_dataset_cols("sensor", check_acc=check_acc, check_gyr=check_gyr),
                    _get_expected_dataset_cols("body", check_acc=check_acc, check_gyr=check_gyr),
                ],
            )
        else:
            _assert_has_columns(dataset, [_get_expected_dataset_cols(frame, check_acc=check_acc, check_gyr=check_gyr)])

    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a SingleSensorDataset. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False
    return True


def is_multi_sensor_dataset(
    dataset: MultiSensorDataset,
    check_acc: bool = True,
    check_gyr: bool = True,
    frame: _ALLOWED_FRAMES_TYPE = "any",
    raise_exception: bool = False,
) -> bool:
    """Check if an object is a valid multi-sensor dataset.

    A valid multi sensor dataset is:

    - is either a :class:`pandas.DataFrame` with 2 level multi-index as columns or a dictionary of single sensor
      datasets (see :func:`~gaitmap.utils.dataset_helper.is_single_sensor_dataset`)

    In case the dataset is a :class:`pandas.DataFrame` with two levels, the first level is expected to be the names
    of the used sensors.
    In both cases (dataframe or dict), `dataset[<sensor_name>]` is expected to return a valid single sensor
    dataset.
    On each of the these single-sensor datasets,
    :func:`~gaitmap.utils.dataset_helper.is_single_sensor_dataset` is used with the same
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
    raise_exception
        If True an exception is raised if the object does not pass the validation.
        If False, the function will return simply True or False.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_single_sensor_dataset: Explanation and checks for single sensor datasets

    """
    try:
        _assert_is_dtype(dataset, (pd.DataFrame, dict))
        if isinstance(dataset, pd.DataFrame):
            _assert_has_multindex_cols(dataset, expected=True, nlevels=2)
        _assert_multisensor_is_not_empty(dataset)
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a MultiSensorDataset. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False

    try:
        for k in get_multi_sensor_dataset_names(dataset):
            is_single_sensor_dataset(
                dataset[k], check_acc=check_acc, check_gyr=check_gyr, frame=frame, raise_exception=True
            )
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object appears to be a MultiSensorDataset, "
                'but for the sensor with the name "{}", the following validation error was raised:\n\n{}'.format(
                    k, str(e)
                )
            ) from e
        return False
    return True


def is_dataset(
    dataset: Dataset, check_acc: bool = True, check_gyr: bool = True, frame: _ALLOWED_FRAMES_TYPE = "any",
) -> Optional[Literal["single", "multi"]]:
    """Check if an object is a valid multi-sensor or single-sensor dataset.

    This function will try to check the input using :func:`~gaitmap.utils.dataset_helper.is_single_sensor_dataset` and
    :func:`~gaitmap.utils.dataset_helper.is_multi_sensor_dataset`.
    In case one of the two checks is successful, a string is returned, which type of dataset the input is.
    Otherwise a descriptive error is raised

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

    Returns
    -------
    dataset_type
        "single" in case of a single-sensor dataset, "multi" in case of a multi-sensor dataset and None in case it is
        neither.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_single_sensor_dataset: Explanation and checks for single sensor datasets
    gaitmap.utils.dataset_helper.is_multi_sensor_dataset: Explanation and checks for multi sensor datasets

    """
    try:
        is_single_sensor_dataset(dataset, check_acc=check_acc, check_gyr=check_gyr, frame=frame, raise_exception=True)
    except ValidationError as e:
        single_error = e
    else:
        return "single"

    try:
        is_multi_sensor_dataset(dataset, check_acc=check_acc, check_gyr=check_gyr, frame=frame, raise_exception=True)
    except ValidationError as e:
        multi_error = e
    else:
        return "multi"

    raise ValidationError(
        "The passed object appears to be neither a single- or a multi-sensor dataset. "
        "Below you can find the errors raised for both checks:\n\n"
        "Single-Sensor\n"
        "=============\n"
        f"{str(single_error)}\n\n"
        "Multi-Sensor\n"
        "=============\n"
        f"{str(multi_error)}"
    )


_ALLOWED_STRIDE_TYPE = Literal["any", "segmented", "min_vel", "ic"]


def is_single_sensor_stride_list(
    stride_list: SingleSensorStrideList, stride_type: _ALLOWED_STRIDE_TYPE = "any", raise_exception: bool = False,
) -> bool:
    """Check if an input is a single-sensor stride list.

    A valid stride list:

    - is a pandas Dataframe with at least the following columns: `["s_id", "start", "end"]`.
      The `s_id` column can also be part of the index.
    - has only a single level column index
    - the value of `s_id` is unique

    Note that this function does only check the structure and not the plausibility of the contained values.
    For this `~gaitmap.utils.stride_list_conversions.enforce_stride_list_consistency` can be used.

    However, depending on the type of stride list, further requirements need to be fulfilled:

    min_vel
        A min-vel stride list describes a stride list that defines a stride from one midstance (`min_vel`) to the next.
        This type of stride list can be performed for ZUPT based trajectory estimation.
        It is expected to additionally have the following columns describing relevant stride events: `["pre_ic", "ic",
        "min_vel", "tc"]`.
        See :mod:`~gaitmap.event_detection` for details.
        For this type of stride list it is further tested, that the "start" column is actual identical to the "min_vel"
        column.
    segmented
        A segmented stride list is a stride list in which every stride starts and ends between min_vel and tc.
        For this stride list, we expect that all relevant events within each stride are already detected.
        Hence, it is checked if columns with the name `["ic", "tc", "min_vel"]` exist.
        If you want to check the structure of a stride list right after the segmentation, where no events are detected
        yet use `"any"` as `stride_type`.
    ic
        A IC stride list is a stride list in which every stride starts and ends with a IC.
        Regarding columns, it has the same requirements as the "segmented" stride list.
        Additionally it is checked, if the "start" columns is actually identical to the "ic" column.

    Parameters
    ----------
    stride_list
        The object that should be tested
    stride_type
        The expected stride type of this object.
        If this is "any" only the generally required columns are checked.
    raise_exception
        If True an exception is raised if the object does not pass the validation.
        If False, the function will return simply True or False.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_multi_sensor_stride_list: Check for multi-sensor stride lists
    gaitmap.utils.stride_list_conversion.enforce_stride_list_consistency: Remove strides that do not have the correct
        event order

    """
    if stride_type != "any" and stride_type not in SL_ADDITIONAL_COLS:
        raise ValueError(
            'The argument `stride_type` must be "any" or one of {}'.format(list(SL_ADDITIONAL_COLS.keys()))
        )

    try:
        _assert_is_dtype(stride_list, pd.DataFrame)
        _assert_has_multindex_cols(stride_list, expected=False)

        stride_list = set_correct_index(stride_list, SL_INDEX)

        # Check if it has the correct columns
        all_columns = [*SL_COLS, *SL_ADDITIONAL_COLS.get(stride_type, [])]
        _assert_has_columns(stride_list, [all_columns])

        start_event = {"min_vel": "min_vel", "ic": "ic"}
        # Check that the start time corresponds to the correct event
        if (
            start_event.get(stride_type, False)
            and len(stride_list) > 0
            and not np.array_equal(stride_list["start"].to_numpy(), stride_list[start_event[stride_type]].to_numpy())
        ):
            raise ValidationError(
                "For a {} stride list, the start column is expected to be identical to the {} column, "
                "but they are different.".format(stride_type, start_event[stride_type])
            )
        # Check that the stride ids are unique
        if not stride_list.index.nunique() == stride_list.index.size:
            raise ValidationError("The stride id of the stride list is expected to be unique.")

    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a SingleSensorStrideList. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False
    return True


def is_multi_sensor_stride_list(
    stride_list: MultiSensorStrideList, stride_type: _ALLOWED_STRIDE_TYPE = "any", raise_exception: bool = False,
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
    raise_exception
        If True an exception is raised if the object does not pass the validation.
        If False, the function will return simply True or False.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_single_sensor_stride_list: Check for multi-sensor stride lists
    gaitmap.utils.stride_list_conversion.enforce_stride_list_consistency: Remove strides that do not have the correct
        event order

    """
    try:
        _assert_is_dtype(stride_list, dict)
        _assert_multisensor_is_not_empty(stride_list)
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a MultiSensorStrideList. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False

    try:
        for k in stride_list.keys():
            is_single_sensor_stride_list(stride_list[k], stride_type=stride_type, raise_exception=True)
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object appears to be a MultiSensorStrideList, "
                'but for the sensor with the name "{}", the following validation error was raised:\n\n{}'.format(
                    k, str(e)
                )
            ) from e
        return False
    return True


def is_stride_list(
    stride_list: StrideList, stride_type: _ALLOWED_STRIDE_TYPE = "any",
) -> Optional[Literal["single", "multi"]]:
    """Check if an object is a valid multi-sensor or single-sensor stride list.

    This function will try to check the input using
    :func:`~gaitmap.utils.dataset_helper.is_single_sensor_stride_list` and
    :func:`~gaitmap.utils.dataset_helper.is_multi_sensor_stride_list`.
    In case one of the two checks is successful, a string is returned, which type of dataset the input is.
    Otherwise a descriptive error is raised

    Parameters
    ----------
    stride_list
        The object that should be tested
    stride_type
        The expected stride type of this object.
        If this is "any" only the generally required columns are checked.

    Returns
    -------
    dataset_type
        "single" in case of a single-sensor stride list, "multi" in case of a multi-sensor stride list and None in case
        it is neither.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_single_sensor_stride_list: Explanation and checks for single sensor stride lists
    gaitmap.utils.dataset_helper.is_multi_sensor_stride_list: Explanation and checks for multi sensor stride lists

    """
    try:
        is_single_sensor_stride_list(stride_list, stride_type=stride_type, raise_exception=True)
    except ValidationError as e:
        single_error = e
    else:
        return "single"

    try:
        is_multi_sensor_stride_list(stride_list, stride_type=stride_type, raise_exception=True)
    except ValidationError as e:
        multi_error = e
    else:
        return "multi"

    raise ValidationError(
        "The passed object appears to be neither a single- or a multi-sensor stride list. "
        "Below you can find the errors raised for both checks:\n\n"
        "Single-Sensor\n"
        "=============\n"
        f"{str(single_error)}\n\n"
        "Multi-Sensor\n"
        "=============\n"
        f"{str(multi_error)}"
    )


def get_single_sensor_regions_of_interest_types(roi_list: SingleSensorRegionsOfInterestList) -> Literal["roi", "gs"]:
    """Identify which type of region of interest list is passed by checking the existing columns."""
    roi_list_columns = roi_list.reset_index().columns
    valid_index_dict = ROI_ID_COLS
    matched_index_col = [col for col in roi_list_columns if col in valid_index_dict.values()]
    if not matched_index_col:
        raise ValidationError(
            "The region of interest list is expected to have one of {} either as a column or in the "
            "index".format(list(valid_index_dict.values()))
        )
    region_type = list(valid_index_dict.keys())[list(valid_index_dict.values()).index(matched_index_col[0])]
    return region_type


def is_single_sensor_regions_of_interest_list(
    roi_list: SingleSensorRegionsOfInterestList,
    region_type: Literal["any", "roi", "gs"] = "any",
    raise_exception: bool = False,
) -> bool:
    """Check if an input is a single-sensor regions-of-interest list.

    A valid region of interest list:

    - is a pandas Dataframe with at least the following columns: `["start", "end"]`
    - additionally it has either a column or a single index named either "roi_id" or "gs_id" depended on the specified
      `region_type`. The value of this id column must be unique.
    - has only a single level column index

    Note that this function does only check the structure and not the plausibility of the contained values.

    Parameters
    ----------
    roi_list
        The object that should be tested
    region_type
        The expected region type of this object.
        If this is "any" any of the possible versions are checked
    raise_exception
        If True an exception is raised if the object does not pass the validation.
        If False, the function will return simply True or False.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_multi_sensor_regions_of_interest_list: Check for multi-sensor regions-of-interest
        lists

    """
    if region_type != "any" and region_type not in ROI_ID_COLS:
        raise ValueError('The argument `region_type` must be "any" or one of {}'.format(list(ROI_ID_COLS.keys())))

    try:
        _assert_is_dtype(roi_list, pd.DataFrame)
        _assert_has_multindex_cols(roi_list, expected=False)

        actual_region_type = get_single_sensor_regions_of_interest_types(roi_list)
        if region_type is not "any" and actual_region_type != region_type:
            raise ValidationError(
                "A ROI list of type {} is expected to have a either an index or a column named {}. "
                "The provided ROI list appears to be of the type {} instead.".format(
                    region_type, ROI_ID_COLS[region_type], actual_region_type
                )
            )

        roi_list = set_correct_index(roi_list, [ROI_ID_COLS[actual_region_type]])
        _assert_has_columns(roi_list, [["start", "end"]])

        # Check that the roi ids are unique
        if not roi_list.index.nunique() == roi_list.index.size:
            raise ValidationError("The roi/gs id of the stride list is expected to be unique.")
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a SingleSensorRegionsOfInterestList. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False

    return True


def is_multi_sensor_regions_of_interest_list(
    roi_list: MultiSensorRegionsOfInterestList,
    region_type: Literal["any", "roi", "gsd"] = "any",
    raise_exception: bool = False,
) -> bool:
    """Check if an input is a multi-sensor stride list.

    A valid multi-sensor stride list is dictionary of single-sensor stride lists.

    This function :func:`~gaitmap.utils.dataset_helper.is_single_sensor_stride_list` for each of the contained stride
    lists.

    Parameters
    ----------
    roi_list
        The object that should be tested
    region_type
        The expected region type of this object.
        If this is "any" any of the possible versions are checked
    raise_exception
        If True an exception is raised if the object does not pass the validation.
        If False, the function will return simply True or False.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_single_sensor_regions_of_interest_list: Check for multi-sensor roi lists

    """
    try:
        _assert_is_dtype(roi_list, dict)
        _assert_multisensor_is_not_empty(roi_list)
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a MultiSensorStrideList. "
                "The validation failed with the following error:\n\n{}".format(str(e))
            ) from e
        return False

    try:
        for k in roi_list.keys():
            is_single_sensor_regions_of_interest_list(roi_list[k], region_type=region_type, raise_exception=True)
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object appears to be a MultiSensorRegionsOfInterestList, "
                'but for the sensor with the name "{}", the following validation error was raised:\n\n{}'.format(
                    k, str(e)
                )
            ) from e
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


def is_single_sensor_position_list(position_list: SingleSensorPositionList, s_id: bool = True) -> bool:
    """Check if an input is a single-sensor position list.

    A valid position list:

    - Is a pandas DataFrame with at least the following columns: `["sample", "pos_x", "pos_y", "pos_z"]`
    - The additional column `["s_id"]` is expected for stride level orientation lists.
    - The columns `sample` (and `s_id`) can also be part of the index instead

    Parameters
    ----------
    position_list
        The object that should be tested
    s_id
        If True the orientation list is expected to have a column (either regular or in the index) that is named `s_id`.
        If False this check is not performed.
        A orientation list without an `s_id` would be used to store orientation information without a clear association
        to a stride.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_multi_sensor_position_list: Check for multi-sensor position lists

    """
    if not isinstance(position_list, pd.DataFrame):
        return False
    index_cols = ["sample"]
    if s_id is True:
        index_cols.append("s_id")
    try:
        position_list = set_correct_index(position_list, index_cols)
    except ValidationError:
        return False
    columns = position_list.columns
    expected_columns = GF_POS
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


def is_single_sensor_velocity_list(velocity_list: SingleSensorVelocityList, s_id: bool = True) -> bool:
    """Check if an input is a single-sensor velocity list.

    A valid position list:

    - Is a pandas DataFrame with at least the following columns: `["sample", "vel_x", "vel_y", "vel_z"]`
    - The additional column `["s_id"]` is expected for stride level orientation lists.
    - The columns `sample` (and `s_id`) can also be part of the index instead

    Parameters
    ----------
    velocity_list
        The object that should be tested
    s_id
        If True the orientation list is expected to have a column (either regular or in the index) that is named `s_id`.
        If False this check is not performed.
        A orientation list without an `s_id` would be used to store orientation information without a clear association
        to a stride.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_multi_sensor_velocity_list: Check for multi-sensor velocity lists

    """
    if not isinstance(velocity_list, pd.DataFrame):
        return False
    index_cols = ["sample"]
    if s_id is True:
        index_cols.append("s_id")
    try:
        velocity_list = set_correct_index(velocity_list, index_cols)
    except ValidationError:
        return False
    columns = velocity_list.columns
    expected_columns = GF_VEL
    if not all(v in columns for v in expected_columns):
        return False
    return True


def is_multi_sensor_velocity_list(velocity_list: MultiSensorVelocityList) -> bool:
    """Check if an input is a multi-sensor velocity list.

    A valid multi-sensor stride list is dictionary of single-sensor velocity lists.

    This function :func:`~gaitmap.utils.dataset_helper.is_single_sensor_velocity_list` for each of the contained stride
    lists.

    Parameters
    ----------
    velocity_list
        The object that should be tested

    See Also
    --------
    gaitmap.utils.dataset_helper.is_single_sensor_velocity_list: Check for multi-sensor velocity lists

    """
    if not isinstance(velocity_list, dict):
        return False

    keys = velocity_list.keys()

    if len(keys) == 0:
        return False

    for k in keys:
        if not is_single_sensor_velocity_list(velocity_list[k]):
            return False
    return True


def is_single_sensor_orientation_list(orientation_list: SingleSensorOrientationList, s_id: bool = True) -> bool:
    """Check if an input is a single-sensor orientation list.

    A valid orientation list:

    - Is a pandas DataFrame with at least the following columns: `["sample", "q_x", "q_y", "q_z", "q_w"]`
    - The additional column `["s_id"]` is expected for stride level orientation lists.
    - The columns `sample` (and `s_id`) can also be part of the index instead


    Parameters
    ----------
    orientation_list
        The object that should be tested
    s_id
        If True the orientation list is expected to have a column (either regular or in the index) that is named `s_id`.
        If False this check is not performed.
        A orientation list without an `s_id` would be used to store orientation information without a clear association
        to a stride.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_multi_sensor_orientation_list: Check for multi-sensor orientation lists

    """
    if not isinstance(orientation_list, pd.DataFrame):
        return False
    index_cols = ["sample"]
    if s_id is True:
        index_cols.append("s_id")
    try:
        orientation_list = set_correct_index(orientation_list, index_cols)
    except ValidationError:
        return False
    columns = orientation_list.columns
    expected_columns = GF_ORI
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
    if not isinstance(orientation_list, dict):
        return False

    keys = orientation_list.keys()

    if len(keys) == 0:
        return False

    for k in keys:
        if not is_single_sensor_orientation_list(orientation_list[k]):
            return False
    return True


def set_correct_index(
    df: pd.DataFrame, index_cols: Iterable[Hashable], drop_false_index_cols: bool = True
) -> pd.DataFrame:
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

    Returns
    -------
    df
        A dataframe with the correct columns set as index

    """
    index_cols = list(index_cols)
    try:
        _assert_has_index_columns(df, index_cols)
        return df
    except ValidationError:
        pass

    # In case not all columns are in the the index, reset_the index and check the column names
    wrong_index = [i for i, n in enumerate(df.index.names) if n not in index_cols]
    all_wrong = len(wrong_index) == len(df.index.names)
    df_just_right_index = df.reset_index(level=wrong_index, drop=drop_false_index_cols)
    if not all_wrong:
        # In case correct indix cols are remaining make them to regular columns
        df_just_right_index = df_just_right_index.reset_index()

    try:
        _assert_has_columns(df_just_right_index, [index_cols])
    except ValidationError as e:
        raise ValidationError(
            "The dataframe is expected to have the following columns either in the index or as columns ({}), "
            "but it has {}".format(index_cols, df.columns)
        ) from e

    return df_just_right_index.set_index(index_cols)
