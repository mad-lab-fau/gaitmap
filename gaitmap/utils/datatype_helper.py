"""A couple of helper functions that easy the use of the typical gaitmap data formats."""

from collections.abc import Iterable, Sequence
from typing import Callable, Optional, Union, cast

import numpy as np
import pandas as pd
from typing_extensions import Literal

from gaitmap.utils._datatype_validation_helper import (
    _ALLOWED_FRAMES,
    _ALLOWED_FRAMES_TYPE,
    _ALLOWED_STRIDE_TYPE,
    _ALLOWED_TRAJ_LIST_TYPES,
    _assert_has_columns,
    _assert_has_index_columns,
    _assert_has_multindex_cols,
    _assert_is_dtype,
    _assert_multisensor_is_not_empty,
    _get_expected_dataset_cols,
    _get_multi_sensor_data_names,
)
from gaitmap.utils._types import _DataFrame, _Hashable
from gaitmap.utils.consts import (
    GF_ORI,
    GF_POS,
    GF_VEL,
    ROI_ID_COLS,
    SL_ADDITIONAL_COLS,
    SL_COLS,
    SL_INDEX,
    SL_MINIMAL_COLS,
    TRAJ_TYPE_COLS,
)
from gaitmap.utils.exceptions import ValidationError

SingleSensorData = _DataFrame
SingleSensorStrideList = _DataFrame
SingleSensorRegionsOfInterestList = _DataFrame
SingleSensorPositionList = _DataFrame
SingleSensorVelocityList = _DataFrame
SingleSensorOrientationList = _DataFrame

MultiSensorData = Union[pd.DataFrame, dict[_Hashable, SingleSensorData]]
SensorData = Union[SingleSensorData, MultiSensorData]

MultiSensorStrideList = dict[_Hashable, pd.DataFrame]
StrideList = Union[SingleSensorStrideList, MultiSensorStrideList]

MultiSensorRegionsOfInterestList = dict[_Hashable, pd.DataFrame]
RegionsOfInterestList = Union[SingleSensorRegionsOfInterestList, MultiSensorRegionsOfInterestList]

MultiSensorPositionList = dict[_Hashable, pd.DataFrame]
PositionList = Union[SingleSensorPositionList, MultiSensorPositionList]

MultiSensorVelocityList = dict[_Hashable, pd.DataFrame]
VelocityList = Union[SingleSensorVelocityList, MultiSensorVelocityList]

MultiSensorOrientationList = dict[_Hashable, pd.DataFrame]
OrientationList = Union[SingleSensorOrientationList, MultiSensorOrientationList]


def to_dict_multi_sensor_data(sensordata: MultiSensorData) -> dict[_Hashable, SingleSensorData]:
    """Convert a multi-sensor data to a dictionary of single sensor datas.

    If it is already in this format, the function will not do anything.
    """
    is_multi_sensor_data(sensordata, check_acc=False, check_gyr=False, raise_exception=True)
    if isinstance(sensordata, dict):
        return sensordata
    return {sensor: sensordata[sensor] for sensor in sensordata.columns.levels[0]}


def is_single_sensor_data(
    data: SingleSensorData,
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
    data
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
    gaitmap.utils.dataset_helper.is_multi_sensor_data: Explanation and checks for multi sensor datasets

    """
    if frame not in _ALLOWED_FRAMES:
        raise ValueError(f"The argument `frame` must be one of {_ALLOWED_FRAMES}")
    try:
        _assert_is_dtype(data, pd.DataFrame)
        _assert_has_multindex_cols(data, expected=False)

        if frame == "any":
            _assert_has_columns(
                data,
                [
                    _get_expected_dataset_cols("sensor", check_acc=check_acc, check_gyr=check_gyr),
                    _get_expected_dataset_cols("body", check_acc=check_acc, check_gyr=check_gyr),
                ],
            )
        else:
            _assert_has_columns(data, [_get_expected_dataset_cols(frame, check_acc=check_acc, check_gyr=check_gyr)])

    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be SingleSensorData. "
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_multi_sensor_data(
    data: MultiSensorData,
    check_acc: bool = True,
    check_gyr: bool = True,
    frame: _ALLOWED_FRAMES_TYPE = "any",
    raise_exception: bool = False,
) -> bool:
    """Check if an object is a valid multi-sensor data object.

    Valid multi sensor data is:

    - is either a :class:`pandas.DataFrame` with 2 level multi-index as columns or a dictionary of single sensor
      datasets (see :func:`~gaitmap.utils.dataset_helper.is_single_sensor_data`)

    In case the data is a :class:`pandas.DataFrame` with two levels, the first level is expected to be the names
    of the used sensors.
    In both cases (dataframe or dict), `dataset[<sensor_name>]` is expected to return a valid single sensor
    dataset.
    On each of the these single-sensor datasets, :func:`~gaitmap.utils.dataset_helper.is_single_sensor_data` is used
    with the same parameters that are used to call this function.

    Parameters
    ----------
    data
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
    gaitmap.utils.dataset_helper.is_single_sensor_data: Explanation and checks for single sensor data

    """
    try:
        _assert_is_dtype(data, (pd.DataFrame, dict))
        if isinstance(data, pd.DataFrame):
            _assert_has_multindex_cols(data, expected=True, nlevels=2)
        _assert_multisensor_is_not_empty(data)
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be MultiSensorData. "
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False

    try:
        for k in get_multi_sensor_names(data):
            is_single_sensor_data(data[k], check_acc=check_acc, check_gyr=check_gyr, frame=frame, raise_exception=True)
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object appears to be MultiSensorData, "
                f'but for the sensor with the name "{k}", the following validation error was raised:\n\n{e!s}'
            ) from e
        return False
    return True


def is_sensor_data(
    data: SensorData, check_acc: bool = True, check_gyr: bool = True, frame: _ALLOWED_FRAMES_TYPE = "any"
) -> Literal["single", "multi"]:
    """Check if an object is valid multi-sensor or single-sensor data.

    This function will try to check the input using :func:`~gaitmap.utils.dataset_helper.is_single_sensor_data` and
    :func:`~gaitmap.utils.dataset_helper.is_multi_sensor_data`.
    In case one of the two checks is successful, a string is returned, which type of dataset the input is.
    Otherwise a descriptive error is raised

    Parameters
    ----------
    data
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
    data_type
        "single" in case of single-sensor data, "multi" in case of multi-sensor data.
        In case it is neither, a ValidationError is raised.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_single_sensor_data: Explanation and checks for single sensor data
    gaitmap.utils.dataset_helper.is_multi_sensor_data: Explanation and checks for multi sensor data

    """
    try:
        is_single_sensor_data(data, check_acc=check_acc, check_gyr=check_gyr, frame=frame, raise_exception=True)
    except ValidationError as e:
        single_error = e
    else:
        return "single"

    try:
        is_multi_sensor_data(data, check_acc=check_acc, check_gyr=check_gyr, frame=frame, raise_exception=True)
    except ValidationError as e:
        multi_error = e
    else:
        return "multi"

    raise ValidationError(
        "The passed object appears to be neither single- or multi-sensor data. "
        "Below you can find the errors raised for both checks:\n\n"
        "Single-Sensor\n"
        "=============\n"
        f"{single_error!s}\n\n"
        "Multi-Sensor\n"
        "=============\n"
        f"{multi_error!s}"
    )


def is_single_sensor_stride_list(
    stride_list: SingleSensorStrideList,
    stride_type: _ALLOWED_STRIDE_TYPE = "any",
    check_additional_cols: Union[bool, tuple[str, ...]] = True,
    raise_exception: bool = False,
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
        Additionally, it is checked, if the "start" columns is actually identical to the "ic" column.

    Parameters
    ----------
    stride_list
        The object that should be tested
    stride_type
        The expected stride type of this object.
        If this is "any" only the generally required columns are checked.
    check_additional_cols
        If True, the additional columns for the given `stride_type` are checked.
        If False, only the minimal required columns for a certain stride_type are checked.
        If a tuple of strings is passed, these columns are additionally checked to the minimal columns.
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
        raise ValueError(f'The argument `stride_type` must be "any" or one of {list(SL_ADDITIONAL_COLS.keys())}')

    try:
        _assert_is_dtype(stride_list, pd.DataFrame)
        _assert_has_multindex_cols(stride_list, expected=False)

        stride_list = set_correct_index(stride_list, SL_INDEX)

        if check_additional_cols is True:
            additional_cols = SL_ADDITIONAL_COLS.get(stride_type, [])
        elif check_additional_cols is False:
            additional_cols = ()
        else:
            additional_cols = check_additional_cols

        # Check if it has the correct columns
        all_columns = [*SL_COLS, *SL_MINIMAL_COLS.get(stride_type, []), *additional_cols]
        _assert_has_columns(stride_list, [all_columns])

        start_event = {"min_vel": "min_vel", "ic": "ic"}
        # Check that the start time corresponds to the correct event
        if (
            start_event.get(stride_type, False)
            and len(stride_list) > 0
            and not np.array_equal(stride_list["start"].to_numpy(), stride_list[start_event[stride_type]].to_numpy())
        ):
            raise ValidationError(
                f"For a {stride_type} stride list, the start column is expected to be identical to the "
                f"{start_event[stride_type]} column, but they are different."
            )
        # Check that the stride ids are unique
        if not stride_list.index.nunique() == stride_list.index.size:
            raise ValidationError("The stride id of the stride list is expected to be unique.")

    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object does not seem to be a SingleSensorStrideList. "
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def is_multi_sensor_stride_list(
    stride_list: MultiSensorStrideList,
    stride_type: _ALLOWED_STRIDE_TYPE = "any",
    check_additional_cols: Union[bool, tuple[str, ...]] = True,
    raise_exception: bool = False,
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
    check_additional_cols
        If True, the additional columns for the given `stride_type` are checked.
        If False, only the minimal required columns are checked.
        If a tuple of strings is passed, these columns are additionally checked to the minimal columns.
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
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False

    try:
        for k in stride_list:
            is_single_sensor_stride_list(
                stride_list[k],
                stride_type=stride_type,
                check_additional_cols=check_additional_cols,
                raise_exception=True,
            )
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object appears to be a MultiSensorStrideList, "
                f'but for the sensor with the name "{k}", the following validation error was raised:\n\n{e!s}'
            ) from e
        return False
    return True


def is_stride_list(
    stride_list: StrideList,
    stride_type: _ALLOWED_STRIDE_TYPE = "any",
    check_additional_cols: Union[bool, tuple[str, ...]] = True,
) -> Literal["single", "multi"]:
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
    check_additional_cols
        If True, the additional columns for the given `stride_type` are checked.
        If False, only the minimal required columns are checked.
        If a tuple of strings is passed, these columns are additionally checked to the minimal columns.

    Returns
    -------
    dataset_type
        "single" in case of a single-sensor stride list, "multi" in case of a multi-sensor stride list.
        In case it is neither, a ValidationError is raised.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_single_sensor_stride_list: Explanation and checks for single sensor stride lists
    gaitmap.utils.dataset_helper.is_multi_sensor_stride_list: Explanation and checks for multi sensor stride lists

    """
    try:
        is_single_sensor_stride_list(
            stride_list, stride_type=stride_type, check_additional_cols=check_additional_cols, raise_exception=True
        )
    except ValidationError as e:
        single_error = e
    else:
        return "single"

    try:
        is_multi_sensor_stride_list(
            stride_list, stride_type=stride_type, check_additional_cols=check_additional_cols, raise_exception=True
        )
    except ValidationError as e:
        multi_error = e
    else:
        return "multi"

    raise ValidationError(
        "The passed object appears to be neither a single- or a multi-sensor stride list. "
        "Below you can find the errors raised for both checks:\n\n"
        "Single-Sensor\n"
        "=============\n"
        f"{single_error!s}\n\n"
        "Multi-Sensor\n"
        "=============\n"
        f"{multi_error!s}"
    )


def get_single_sensor_regions_of_interest_types(roi_list: SingleSensorRegionsOfInterestList) -> Literal["roi", "gs"]:
    """Identify which type of region of interest list is passed by checking the existing columns."""
    roi_list_columns = roi_list.reset_index().columns
    valid_index_dict = ROI_ID_COLS
    matched_index_col = [col for col in roi_list_columns if col in valid_index_dict.values()]
    if not matched_index_col:
        raise ValidationError(
            f"The region of interest list is expected to have one of {list(valid_index_dict.values())} either as a "
            "column or in the index"
        )
    region_type = cast(
        Literal["roi", "gs"], list(valid_index_dict.keys())[list(valid_index_dict.values()).index(matched_index_col[0])]
    )
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
        raise ValueError(f'The argument `region_type` must be "any" or one of {list(ROI_ID_COLS.keys())}')

    try:
        _assert_is_dtype(roi_list, pd.DataFrame)
        _assert_has_multindex_cols(roi_list, expected=False)

        actual_region_type = get_single_sensor_regions_of_interest_types(roi_list)
        if region_type not in ("any", actual_region_type):
            raise ValidationError(
                f"A ROI list of type {region_type} is expected to have a either an index or a column named "
                f"{ROI_ID_COLS[region_type]}."
                "The provided ROI list appears to be of the type {actual_region_type} instead."
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
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False

    return True


def is_multi_sensor_regions_of_interest_list(
    roi_list: MultiSensorRegionsOfInterestList,
    region_type: Literal["any", "roi", "gs"] = "any",
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
                "The passed object does not seem to be a MultiSensorRegionsOfInterestList. "
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False

    try:
        for k in roi_list:
            is_single_sensor_regions_of_interest_list(roi_list[k], region_type=region_type, raise_exception=True)
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                "The passed object appears to be a MultiSensorRegionsOfInterestList, "
                f'but for the sensor with the name "{k}", the following validation error was raised:\n\n{e!s}'
            ) from e
        return False
    return True


def is_regions_of_interest_list(
    roi_list: RegionsOfInterestList, region_type: Literal["any", "roi", "gs"] = "any"
) -> Literal["single", "multi"]:
    """Check if an object is a valid multi-sensor or single-sensor regions of interest list.

    This function will try to check the input using
    :func:`~gaitmap.utils.dataset_helper.is_single_sensor_regions_of_interest_list` and
    :func:`~gaitmap.utils.dataset_helper.is_multi_sensor_regions_of_interest_list`.
    In case one of the two checks is successful, a string is returned, which type of dataset the input is.
    Otherwise a descriptive error is raised

    Parameters
    ----------
    roi_list
        The object that should be tested
    region_type
        The expected region type of this object.
        If this is "any" any of the possible versions are checked

    Returns
    -------
    dataset_type
        "single" in case of a single-sensor roi list, "multi" in case of a multi-sensor roi list.
        In case it is neither, an error is raised.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_single_sensor_regions_of_interest_list: Explanation and checks for single sensor
        regions of interest list
    gaitmap.utils.dataset_helper.is_multi_sensor_regions_of_interest_list: Explanation and checks for multi sensor
        regions of interest list

    """
    try:
        is_single_sensor_regions_of_interest_list(roi_list, region_type=region_type, raise_exception=True)
    except ValidationError as e:
        single_error = e
    else:
        return "single"

    try:
        is_multi_sensor_regions_of_interest_list(roi_list, region_type=region_type, raise_exception=True)
    except ValidationError as e:
        multi_error = e
    else:
        return "multi"

    raise ValidationError(
        "The passed object appears to be neither a single- or a multi-sensor regions of interest list. "
        "Below you can find the errors raised for both checks:\n\n"
        "Single-Sensor\n"
        "=============\n"
        f"{single_error!s}\n\n"
        "Multi-Sensor\n"
        "=============\n"
        f"{multi_error!s}"
    )


def get_multi_sensor_names(dataset: MultiSensorData) -> Sequence[_Hashable]:
    """Get the list of sensor names from a multi-sensor dataset.

    .. warning:
        This will not check, if the input is actually a multi-sensor dataset

    Notes
    -----
    The keys are not guaranteed to be ordered.

    """
    return _get_multi_sensor_data_names(dataset=dataset)


def get_single_sensor_trajectory_list_types(
    traj_list: Union[SingleSensorPositionList, SingleSensorOrientationList, SingleSensorVelocityList],
) -> Literal["roi", "gs", "stride"]:
    """Identify which type of trajectory list is passed by checking the existing columns."""
    traj_list_columns = traj_list.reset_index().columns
    valid_index_dict = TRAJ_TYPE_COLS
    matched_index_col = [col for col in traj_list_columns if col in valid_index_dict.values()]
    if not matched_index_col:
        raise ValidationError(
            "The trajectory (orientation, position, velocity) list is expected to have one of "
            f"{list(valid_index_dict.values())} either as a column or in the index"
        )
    list_type = cast(
        Literal["roi", "gs", "stride"],
        list(valid_index_dict.keys())[list(valid_index_dict.values()).index(matched_index_col[0])],
    )
    return list_type


def _is_single_sensor_trajectory_list(
    input_prefix: str,
    input_datatype: str,
    expected_cols: list[str],
    traj_list: Union[SingleSensorOrientationList, SingleSensorVelocityList, SingleSensorOrientationList],
    traj_list_type: Optional[_ALLOWED_TRAJ_LIST_TYPES] = None,
    raise_exception: bool = False,
) -> bool:
    if traj_list_type and traj_list_type != "any_roi" and traj_list_type not in TRAJ_TYPE_COLS:
        raise ValueError(
            "The argument `{}_type` must be None, or one of {} and not {}.".format(
                input_prefix, [*list(ROI_ID_COLS.keys()), "any_roi"], traj_list_type
            )
        )
    try:
        _assert_is_dtype(traj_list, pd.DataFrame)
        _assert_has_multindex_cols(traj_list, expected=False)
        traj_list = set_correct_index(traj_list, ["sample"], drop_false_index_cols=False)
        if traj_list_type == "any_roi":
            nested_expected_cols = [[*expected_cols, TRAJ_TYPE_COLS["roi"]], [*expected_cols, TRAJ_TYPE_COLS["gs"]]]
        elif traj_list_type:
            nested_expected_cols = [[*expected_cols, TRAJ_TYPE_COLS[traj_list_type]]]
        else:
            nested_expected_cols = [expected_cols]
        _assert_has_columns(traj_list, nested_expected_cols)
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                f"The passed object does not seem to be a {input_datatype}. "
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False
    return True


def _is_multi_sensor_trajectory_list(
    input_datatype: str,
    single_func: Callable,
    traj_list: Union[MultiSensorOrientationList, MultiSensorVelocityList, MultiSensorOrientationList],
    raise_exception: bool = False,
    **kwargs,
) -> bool:
    try:
        _assert_is_dtype(traj_list, dict)
        _assert_multisensor_is_not_empty(traj_list)
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                f"The passed object does not seem to be a {input_datatype}. "
                f"The validation failed with the following error:\n\n{e!s}"
            ) from e
        return False

    try:
        for k in traj_list:
            single_func(traj_list[k], **kwargs, raise_exception=True)
    except ValidationError as e:
        if raise_exception is True:
            raise ValidationError(
                f"The passed object appears to be a {input_datatype}, "
                f'but for the sensor with the name "{k}", the following validation error was raised:\n\n{e!s}'
            ) from e
        return False
    return True


def _is_trajectory_list(
    input_datatype: str,
    single_func: Callable,
    multi_func: Callable,
    traj_list: Union[MultiSensorOrientationList, MultiSensorVelocityList, MultiSensorOrientationList],
    **kwargs,
) -> Literal["single", "multi"]:
    try:
        single_func(traj_list, **kwargs, raise_exception=True)
    except ValidationError as e:
        single_error = e
    else:
        return "single"

    try:
        multi_func(traj_list, **kwargs, raise_exception=True)
    except ValidationError as e:
        multi_error = e
    else:
        return "multi"

    raise ValidationError(
        f"The passed object appears to be neither a single- or a multi-sensor {input_datatype}. "
        "Below you can find the errors raised for both checks:\n\n"
        "Single-Sensor\n"
        "=============\n"
        f"{single_error!s}\n\n"
        "Multi-Sensor\n"
        "=============\n"
        f"{multi_error!s}"
    )


def is_single_sensor_position_list(
    position_list: SingleSensorPositionList,
    position_list_type: Optional[_ALLOWED_TRAJ_LIST_TYPES] = None,
    raise_exception: bool = False,
) -> bool:
    """Check if an input is a single-sensor position list.

    A valid position list:

    - Is a pandas DataFrame with at least the following columns: `["sample", "pos_x", "pos_y", "pos_z"]`
    - The additional column `"s_id"`, `"roi_id"`, or `"gs_id"` is expected for stride, roi, and gait sequence (gs)
      level position lists, respectively.
      In case of "any_roi" the name of this column can be either `"roi_id"` or `"gs_id"`.
    - This column and the `sample` column can also be part of the index instead

    Parameters
    ----------
    position_list
        The object that should be tested
    position_list_type
        Which type of index to expect.
        If it is None, only a "sample" column is expected.
        In the remaining cases, an additional column with the correct name is expected.
        See the function description for details.
    raise_exception
        If True an exception is raised if the object does not pass the validation.
        If False, the function will return simply True or False.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_multi_sensor_position_list: Check for multi-sensor position lists

    """
    return _is_single_sensor_trajectory_list(
        input_prefix="position",
        input_datatype="SingleSensorPositionList",
        expected_cols=GF_POS,
        traj_list=position_list,
        traj_list_type=position_list_type,
        raise_exception=raise_exception,
    )


def is_multi_sensor_position_list(
    position_list: MultiSensorPositionList,
    position_list_type: Optional[_ALLOWED_TRAJ_LIST_TYPES] = None,
    raise_exception: bool = False,
) -> bool:
    """Check if an input is a multi-sensor position list.

    A valid multi-sensor stride list is dictionary of single-sensor position lists.

    This function :func:`~gaitmap.utils.dataset_helper.is_single_sensor_position_list` for each of the contained stride
    lists.

    Parameters
    ----------
    position_list
        The object that should be tested
    position_list_type
        Which type of index to expect.
        If it is None, only a "sample" column is expected.
        In the remaining cases, an additional column with the correct name is expected.
        See the function description for details.
    raise_exception
        If True an exception is raised if the object does not pass the validation.
        If False, the function will return simply True or False.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_single_sensor_position_list: Check for multi-sensor position lists

    """
    return _is_multi_sensor_trajectory_list(
        "MultiSensorPositionList",
        is_single_sensor_position_list,
        traj_list=position_list,
        raise_exception=raise_exception,
        position_list_type=position_list_type,
    )


def is_position_list(position_list: PositionList, position_list_type: Optional[_ALLOWED_TRAJ_LIST_TYPES] = None):
    """Check if an object is a valid multi-sensor or single-sensor position list.

    This function will try to check the input using
    :func:`~gaitmap.utils.dataset_helper.is_single_sensor_position_list` and
    :func:`~gaitmap.utils.dataset_helper.is_multi_sensor_position_list`.
    In case one of the two checks is successful, a string is returned, which type of dataset the input is.
    Otherwise a descriptive error is raised

    Parameters
    ----------
    position_list
        The object that should be tested
    position_list_type
        The expected position list type of this object.
        If this is "any" any of the possible versions are checked

    Returns
    -------
    dataset_type
        "single" in case of a single-sensor position_list, "multi" in case of a multi-sensor position_list.
        In case it is neither a ValidationError is raised.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_single_sensor_position_list: Explanation and checks for single sensor
        position list
    gaitmap.utils.dataset_helper.is_multi_sensor_position_list: Explanation and checks for multi sensor
        position list

    """
    return _is_trajectory_list(
        "position list",
        is_single_sensor_position_list,
        is_multi_sensor_position_list,
        traj_list=position_list,
        position_list_type=position_list_type,
    )


def is_single_sensor_velocity_list(
    velocity_list: SingleSensorVelocityList,
    velocity_list_type: Optional[_ALLOWED_TRAJ_LIST_TYPES] = None,
    raise_exception: bool = False,
) -> bool:
    """Check if an input is a single-sensor velocity list.

    A valid velocity list:

    - Is a pandas DataFrame with at least the following columns: `["sample", "vel_x", "vel_y", "vel_z"]`
    - The additional column `"s_id"`, `"roi_id"`, or `"gs_id"` is expected for stride, roi, and gait sequence (gs)
      level velocity lists, respectively.
      In case of "any_roi" the name of this column can be either `"roi_id"` or `"gs_id"`.
    - This column and the `sample` column can also be part of the index instead

    Parameters
    ----------
    velocity_list
        The object that should be tested
    velocity_list_type
        Which type of index to expect.
        If it is None, only a "sample" column is expected.
        In the remaining cases, an additional column with the correct name is expected.
        See the function description for details.
    raise_exception
        If True an exception is raised if the object does not pass the validation.
        If False, the function will return simply True or False.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_multi_sensor_velocity_list: Check for multi-sensor velocity lists

    """
    return _is_single_sensor_trajectory_list(
        input_prefix="velocity",
        input_datatype="SingleSensorVelocityList",
        expected_cols=GF_VEL,
        traj_list=velocity_list,
        traj_list_type=velocity_list_type,
        raise_exception=raise_exception,
    )


def is_multi_sensor_velocity_list(
    velocity_list: MultiSensorVelocityList,
    velocity_list_type: Optional[_ALLOWED_TRAJ_LIST_TYPES] = None,
    raise_exception: bool = False,
) -> bool:
    """Check if an input is a multi-sensor velocity list.

    A valid multi-sensor stride list is dictionary of single-sensor velocity lists.

    This function :func:`~gaitmap.utils.dataset_helper.is_single_sensor_velocity_list` for each of the contained stride
    lists.

    Parameters
    ----------
    velocity_list
        The object that should be tested
    velocity_list_type
        Which type of index to expect.
        If it is None, only a "sample" column is expected.
        In the remaining cases, an additional column with the correct name is expected.
        See the function description for details.
    raise_exception
        If True an exception is raised if the object does not pass the validation.
        If False, the function will return simply True or False.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_single_sensor_velocity_list: Check for multi-sensor velocity lists

    """
    return _is_multi_sensor_trajectory_list(
        "MultiSensorVelocityList",
        is_single_sensor_velocity_list,
        traj_list=velocity_list,
        raise_exception=raise_exception,
        velocity_list_type=velocity_list_type,
    )


def is_velocity_list(velocity_list: VelocityList, velocity_list_type: Optional[_ALLOWED_TRAJ_LIST_TYPES] = None):
    """Check if an object is a valid multi-sensor or single-sensor velocity list.

    This function will try to check the input using
    :func:`~gaitmap.utils.dataset_helper.is_single_sensor_velocity_list` and
    :func:`~gaitmap.utils.dataset_helper.is_multi_sensor_velocity_list`.
    In case one of the two checks is successful, a string is returned, which type of dataset the input is.
    Otherwise a descriptive error is raised

    Parameters
    ----------
    velocity_list
        The object that should be tested
    velocity_list_type
        The expected velocity list type of this object.
        If this is "any" any of the possible versions are checked

    Returns
    -------
    dataset_type
        "single" in case of a single-sensor velocity_list, "multi" in case of a multi-sensor velocity_list.
        In case it is neither a ValidationError is raised.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_single_sensor_velocity_list: Explanation and checks for single sensor
        velocity list
    gaitmap.utils.dataset_helper.is_multi_sensor_velocity_list: Explanation and checks for multi sensor
        velocity list

    """
    return _is_trajectory_list(
        "velocity list",
        is_single_sensor_velocity_list,
        is_multi_sensor_velocity_list,
        traj_list=velocity_list,
        velocity_list_type=velocity_list_type,
    )


def is_single_sensor_orientation_list(
    orientation_list: SingleSensorOrientationList,
    orientation_list_type: Optional[_ALLOWED_TRAJ_LIST_TYPES] = None,
    raise_exception: bool = False,
) -> bool:
    """Check if an input is a single-sensor orientation list.

    A valid orientation list:

    - Is a pandas DataFrame with at least the following columns: `["sample", "q_x", "q_y", "q_z", "q_w"]`
    - The additional column `"s_id"`, `"roi_id"`, or `"gs_id"` is expected for stride, roi, and gait sequence (gs)
      level orientation lists, respectively.
      In case of "any_roi" the name of this column can be either `"roi_id"` or `"gs_id"`.
    - This column and the `sample` column can also be part of the index instead

    Parameters
    ----------
    orientation_list
        The object that should be tested
    orientation_list_type
        Which type of index to expect.
        If it is None, only a "sample" column is expected.
        In the remaining cases, an additional column with the correct name is expected.
        See the function description for details.
    raise_exception
        If True an exception is raised if the object does not pass the validation.
        If False, the function will return simply True or False.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_multi_sensor_orientation_list: Check for multi-sensor orientation lists

    """
    return _is_single_sensor_trajectory_list(
        input_prefix="orientation",
        input_datatype="SingleSensorOrientationList",
        expected_cols=GF_ORI,
        traj_list=orientation_list,
        traj_list_type=orientation_list_type,
        raise_exception=raise_exception,
    )


def is_multi_sensor_orientation_list(
    orientation_list: MultiSensorOrientationList,
    orientation_list_type: Optional[_ALLOWED_TRAJ_LIST_TYPES] = None,
    raise_exception: bool = False,
) -> bool:
    """Check if an input is a multi-sensor orientation list.

    A valid multi-sensor stride list is dictionary of single-sensor Orientation lists.

    Function :func:`~gaitmap.utils.dataset_helper.is_single_sensor_orientation_list` for each of the contained stride
    lists.

    Parameters
    ----------
    orientation_list
        The object that should be tested
    orientation_list_type
        Which type of index to expect.
        If it is None, only a "sample" column is expected.
        In the remaining cases, an additional column with the correct name is expected.
        See the function description for details.
    raise_exception
        If True an exception is raised if the object does not pass the validation.
        If False, the function will return simply True or False.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_single_sensor_orientation_list: Check for multi-sensor orientation lists

    """
    return _is_multi_sensor_trajectory_list(
        "MultiSensorOrientationList",
        is_single_sensor_orientation_list,
        traj_list=orientation_list,
        raise_exception=raise_exception,
        orientation_list_type=orientation_list_type,
    )


def is_orientation_list(
    orientation_list: OrientationList, orientation_list_type: Optional[_ALLOWED_TRAJ_LIST_TYPES] = None
):
    """Check if an object is a valid multi-sensor or single-sensor orientation list.

    This function will try to check the input using
    :func:`~gaitmap.utils.dataset_helper.is_single_sensor_orientation_list` and
    :func:`~gaitmap.utils.dataset_helper.is_multi_sensor_orientation_list`.
    In case one of the two checks is successful, a string is returned, which type of dataset the input is.
    Otherwise a descriptive error is raised

    Parameters
    ----------
    orientation_list
        The object that should be tested
    orientation_list_type
        The expected orientation list type of this object.
        If this is "any" any of the possible versions are checked

    Returns
    -------
    dataset_type
        "single" in case of a single-sensor orientation_list, "multi" in case of a multi-sensor orientation_list and
        None in
        case it is neither.

    See Also
    --------
    gaitmap.utils.dataset_helper.is_single_sensor_orientation_list: Explanation and checks for single sensor
        orientation list
    gaitmap.utils.dataset_helper.is_multi_sensor_orientation_list: Explanation and checks for multi sensor
        orientation list

    """
    return _is_trajectory_list(
        "orientation list",
        is_single_sensor_orientation_list,
        is_multi_sensor_orientation_list,
        traj_list=orientation_list,
        orientation_list_type=orientation_list_type,
    )


def set_correct_index(
    df: pd.DataFrame, index_cols: Iterable[_Hashable], drop_false_index_cols: bool = True
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
    except ValidationError:
        pass
    else:
        return df

    # In case not all columns are in the index, reset_the index and check the column names
    wrong_index = [i for i, n in enumerate(df.index.names) if n not in index_cols]
    all_wrong = len(wrong_index) == len(df.index.names)
    df_just_right_index = df.reset_index(level=wrong_index, drop=drop_false_index_cols)
    if not all_wrong:
        # In case correct index cols are remaining make them to regular columns
        df_just_right_index = df_just_right_index.reset_index()

    try:
        _assert_has_columns(df_just_right_index, [index_cols])
    except ValidationError as e:
        raise ValidationError(
            "The dataframe is expected to have the following columns either in the index or as columns "
            f"({index_cols}), but it has {df.columns}"
        ) from e

    return df_just_right_index.set_index(index_cols)
