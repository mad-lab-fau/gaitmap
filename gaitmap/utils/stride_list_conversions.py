"""A couple of utils to convert stride lists into different formats."""
from typing_extensions import Literal
import numpy as np

from gaitmap.utils.consts import SL_EVENT_ORDER
from gaitmap.utils.dataset_helper import (
    StrideList,
    SingleSensorStrideList,
    is_single_sensor_stride_list,
    is_multi_sensor_stride_list,
)


def segmented_stride_list_to_min_vel(stride_list: StrideList) -> StrideList:
    if is_single_sensor_stride_list(stride_list):
        return _segmented_stride_list_to_min_vel_single_sensor(stride_list)
    if is_multi_sensor_stride_list(stride_list):
        return {k: _segmented_stride_list_to_min_vel_single_sensor(v) for k, v in stride_list.items()}
    raise ValueError("The provided stride list format is not supported.")


def _segmented_stride_list_to_min_vel_single_sensor(stride_list: SingleSensorStrideList) -> SingleSensorStrideList:
    pass


def enforce_stride_list_consistency(
    stride_list: SingleSensorStrideList, stride_type=Literal["segmented", "min_vel", "ic"], check_stride_list: bool = True
) -> SingleSensorStrideList:
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

    """
    if check_stride_list is True and not is_single_sensor_stride_list(stride_list, stride_type=stride_type):
        raise ValueError("The provided stride list format is not supported.")
    # TODO: Test for all stride list types
    order = SL_EVENT_ORDER[stride_type]
    bool_map = np.logical_and.reduce([stride_list[order[i]] < stride_list[order[i+1]] for i in range(len(order) - 1)])
    stride_list = stride_list[bool_map]

    return stride_list
