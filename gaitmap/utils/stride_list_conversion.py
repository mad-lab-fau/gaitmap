"""A couple of utils to convert stride lists into different formats."""
from typing import Tuple

from typing_extensions import Literal
import numpy as np

from gaitmap.utils.consts import SL_EVENT_ORDER
from gaitmap.utils.dataset_helper import (
    StrideList,
    SingleSensorStrideList,
    is_single_sensor_stride_list,
    is_multi_sensor_stride_list,
)


def convert_segmented_stride_list(stride_list: StrideList, target_stride_type: Literal["min_vel", "ic"]) -> StrideList:

    if is_single_sensor_stride_list(stride_list, stride_type="segmented"):
        return _segmented_stride_list_to_min_vel_single_sensor(stride_list, target_stride_type=target_stride_type)[0]
    if is_multi_sensor_stride_list(stride_list, stride_type="segmented"):
        return {
            k: _segmented_stride_list_to_min_vel_single_sensor(v, target_stride_type=target_stride_type)[0]
            for k, v in stride_list.items()
        }
    raise ValueError("The provided stride list format is not supported.")


def _segmented_stride_list_to_min_vel_single_sensor(
    stride_list: SingleSensorStrideList, target_stride_type: Literal["min_vel", "ic"]
) -> Tuple[SingleSensorStrideList, SingleSensorStrideList]:
    converted_stride_list = stride_list.copy()
    converted_stride_list["old_start"] = converted_stride_list["start"]
    converted_stride_list["old_end"] = converted_stride_list["end"]

    # start of each stride is now the new start event
    converted_stride_list["start"] = converted_stride_list[target_stride_type]
    # end of each stride is now the start event of the next strides
    # Breaks in the stride list will be filtered later
    converted_stride_list["end"] = converted_stride_list[target_stride_type].shift(-1)
    if target_stride_type == "min_vel":
        # pre-ic of each stride is the ic in the current segmented stride
        converted_stride_list["pre_ic"] = converted_stride_list["ic"]
        # ic of each stride is the ic in the subsequent segmented stride
        converted_stride_list["ic"] = converted_stride_list["ic"].shift(-1)
        # tc of each stride is the tc in the subsequent segmented stride
        converted_stride_list["tc"] = converted_stride_list["tc"].shift(-1)

    elif target_stride_type == "ic":
        # As the ic occurs after the tc in the segmented stride, new tc is the tc of the next stride
        converted_stride_list["tc"] = converted_stride_list["tc"].shift(-1)

    # drop remaining nans (last list elements will get some nans by shift(-1) operation above)
    converted_stride_list = converted_stride_list.dropna(how="any")

    # Find breaks in the stride list, which indicate the ends of individual gait sequences.
    breaks = (converted_stride_list["old_end"] - converted_stride_list["old_start"].shift(-1)).fillna(0) != 0

    # drop unneeded tmp columns
    converted_stride_list = converted_stride_list.drop(["old_start", "old_end"], axis=1)

    # Remove the last stride of each gait sequence as its end value is already part of the next gait sequence
    converted_stride_list = converted_stride_list[~breaks]

    return converted_stride_list, stride_list[~stride_list["s_id"].isin(converted_stride_list["s_id"])]


def enforce_stride_list_consistency(
    stride_list: SingleSensorStrideList,
    stride_type=Literal["segmented", "min_vel", "ic"],
    check_stride_list: bool = True,
) -> Tuple[SingleSensorStrideList, SingleSensorStrideList]:
    """Exclude those strides where the gait events do not match the expected order.

    Correct order in depends on the stride type:

    - segmented: ["tc", "ic", "min_vel"]
    - min_vel: ["pre_ic", "min_vel", "tc", "ic"]
    - ic: ["ic", "min_vel", "tc"]

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
    if check_stride_list is True and not is_single_sensor_stride_list(stride_list, stride_type=stride_type):
        raise ValueError("The provided stride list format is not supported.")
    # TODO: Test for all stride list types
    order = SL_EVENT_ORDER[stride_type]
    bool_map = np.logical_and.reduce([stride_list[order[i]] < stride_list[order[i + 1]] for i in range(len(order) - 1)])

    return stride_list[bool_map], stride_list[~bool_map]
