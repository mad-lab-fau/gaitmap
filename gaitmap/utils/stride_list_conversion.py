"""A couple of utils to convert stride lists into different formats."""
from typing import List, Tuple

import numpy as np
import pandas as pd
from typing_extensions import Literal

from gaitmap.utils.consts import SL_EVENT_ORDER, SL_INDEX
from gaitmap.utils.datatype_helper import (
    SingleSensorRegionsOfInterestList,
    SingleSensorStrideList,
    StrideList,
    is_single_sensor_regions_of_interest_list,
    is_single_sensor_stride_list,
    is_stride_list,
    set_correct_index,
)


def convert_segmented_stride_list(stride_list: StrideList, target_stride_type: Literal["min_vel", "ic"],
                                  source_stride_type: Literal["segmented", "ic"] = "segmented") -> StrideList:
    """Convert a segmented stride list with detected events into other types of stride lists.

    During the conversion some strides might be removed.
    For more information about the different types of stride lists see the :ref:`stride list guide <stride_list_guide>`.

    Parameters
    ----------
    stride_list
        Stride list to be converted
    target_stride_type
        The stride list type that should be converted to
    source_stride_type
        The stride list type that should be converted from

    Returns
    -------
    converted_stride_list
        Stride list in the new format

    """
    stride_list_type = is_stride_list(stride_list, stride_type="segmented")
    if stride_list_type == "single":
        return _segmented_stride_list_to_min_vel_single_sensor(stride_list, target_stride_type=target_stride_type,
                                                               source_stride_type=source_stride_type)[0]
    return {
        k: _segmented_stride_list_to_min_vel_single_sensor(v, target_stride_type=target_stride_type,
                                                           source_stride_type=source_stride_type)[0]
        for k, v in stride_list.items()
    }


def _segmented_stride_list_to_min_vel_single_sensor(
        stride_list: SingleSensorStrideList, target_stride_type: Literal["min_vel", "ic"],
        source_stride_type: Literal["segmented", "ic"]
) -> Tuple[SingleSensorStrideList, SingleSensorStrideList]:
    """Convert a segmented stride list with detected events into other types of stride lists.

    During the conversion some strides might be removed.
    For more information about the different types of stride lists see the :ref:`stride list guide <stride_list_guide>`.

    Note, this function does not check if the input is a proper stride list.

    Parameters
    ----------
    stride_list
        Stride list to be converted
    target_stride_type
        The stride list type that should be converted to

    Returns
    -------
    converted_stride_list
        Stride list in the new format
    removed_strides
        Strides that were removed during the conversion.
        This stride list is still in the input format.

    """
    stride_list = set_correct_index(stride_list, SL_INDEX)
    converted_stride_list = stride_list.copy()
    converted_stride_list["old_start"] = converted_stride_list["start"]
    converted_stride_list["old_end"] = converted_stride_list["end"]

    # start of each stride is now the new start event
    converted_stride_list["start"] = converted_stride_list[target_stride_type]
    # end of each stride is now the start event of the next strides
    # Breaks in the stride list will be filtered later
    converted_stride_list["end"] = converted_stride_list[target_stride_type].shift(-1)
    if target_stride_type == "min_vel":
        if "ic" in converted_stride_list.columns:
            # pre-ic of each stride is the ic in the current segmented stride
            converted_stride_list["pre_ic"] = converted_stride_list["ic"]
            # ic of each stride is the ic in the subsequent segmented stride
            converted_stride_list["ic"] = converted_stride_list["ic"].shift(-1)
        if "tc" in converted_stride_list.columns and source_stride_type == "segmented":
            # do not shift if source_stride_type is "ic"
            # tc of each stride is the tc in the subsequent segmented stride
            converted_stride_list["tc"] = converted_stride_list["tc"].shift(-1)

    elif target_stride_type == "ic" and "tc" in converted_stride_list.columns:
        # As the ic occurs after the tc in the segmented stride, new tc is the tc of the next stride
        converted_stride_list["tc"] = converted_stride_list["tc"].shift(-1)

    # Find breaks in the stride list, which indicate the ends of individual gait sequences.
    breaks = (converted_stride_list["old_end"] - converted_stride_list["old_start"].shift(-1)).fillna(0) != 0

    # drop unneeded tmp columns
    converted_stride_list = converted_stride_list.drop(["old_start", "old_end"], axis=1)

    # Remove the last stride of each gait sequence as its end value is already part of the next gait sequence
    converted_stride_list = converted_stride_list[~breaks]

    # drop remaining nans (last list elements will get some nans by shift(-1) operation above)
    converted_stride_list = converted_stride_list.dropna(how="any")

    return converted_stride_list, stride_list[~stride_list.index.isin(converted_stride_list.index)]


def enforce_stride_list_consistency(
        stride_list: SingleSensorStrideList,
        stride_type=Literal["segmented", "min_vel", "ic"],
        check_stride_list: bool = True,
) -> Tuple[SingleSensorStrideList, SingleSensorStrideList]:
    """Exclude those strides where the gait events do not match the expected order or contain NaN.

    Correct order in depends on the stride type:

    - segmented: ["tc", "ic", "min_vel"]
    - min_vel: ["pre_ic", "min_vel", "tc", "ic"]
    - ic: ["ic", "min_vel", "tc"]

    If only a subset of the required events exists, only the order of the existing events is checked.

    Parameters
    ----------
    stride_list
        A single sensor stride list in a Dataframe format
    stride_type
        Indicate which types of strides are expected to be in the stride list.
        This changes the expected columns and order of events.
    check_stride_list
        If True `~gaitmap.utils.dataset_helper.is_single_sensor_stride_list` is used to check the overall format of the
        stride list.
        Setting this to False might be useful if you use the consistency check while you are still building a proper
        stride list.

    Returns
    -------
    cleaned_stride_list
        stride_list but will all invalid strides removed
    invalid_strides
        all strides that were removed

    """
    if check_stride_list is True:
        is_single_sensor_stride_list(stride_list, stride_type=stride_type, raise_exception=True)
    order = SL_EVENT_ORDER[stride_type]
    order = [c for c in order if c in stride_list.columns]

    if len(order) == 0:
        raise ValueError("No valid events found in the stride list.")

    if len(order) == 1:
        # only one event, no need to check order
        return stride_list, pd.DataFrame(columns=stride_list.columns)

    # Note: the following also drops strides that contain NaN for any event
    bool_map = np.logical_and.reduce([stride_list[order[i]] < stride_list[order[i + 1]] for i in range(len(order) - 1)])

    return stride_list[bool_map], stride_list[~bool_map]


def intersect_stride_list(
        stride_event_list: SingleSensorStrideList,
        regions_of_interest: SingleSensorRegionsOfInterestList,
) -> List[SingleSensorStrideList]:
    """Split the stride list into multiple stride lists based on the regions of interest.

    All events in the returned stride lists are made relative to the start of the region of interest.

    In all cases, only strides that are fully contained within a region of interest are included in the output stride
    lists.

    .. warning:: This method substracts the start of the ROI list from all events in the stride list.
                 This means, columns that don't represent events in samples are not supported.

    Parameters
    ----------
    stride_event_list
        The event list to split
    regions_of_interest
        The regions of interest to split the stride list into.
        The ROIs can overlap.
        In this case, strides appear in multiple stride lists.

    Returns
    -------
    stride_lists
        A list of stride lists, one for each region of interest.


    """
    is_single_sensor_stride_list(stride_event_list, raise_exception=True)
    is_single_sensor_regions_of_interest_list(regions_of_interest, raise_exception=True)
    stride_list = set_correct_index(stride_event_list.copy(), SL_INDEX)
    roi_list = regions_of_interest.copy()

    stride_lists = []

    for _, roi in roi_list.iterrows():
        # find all strides that are fully contained in the roi
        partial_stride_list = stride_list.loc[
            (stride_list["start"] >= roi["start"]) & (stride_list["end"] <= roi["end"])
            ]
        partial_stride_list -= roi["start"]
        stride_lists.append(partial_stride_list)

    return stride_lists
