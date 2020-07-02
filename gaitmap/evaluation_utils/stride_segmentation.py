from typing import Union, Tuple

import numpy as np
import pandas as pd

from gaitmap.utils.consts import SL_INDEX
from gaitmap.utils.dataset_helper import StrideList, set_correct_index, is_single_sensor_stride_list


def evaluate_segmented_stride_list(
    segmented_stride_list: StrideList,
    ground_truth: StrideList,
    tolerance: Union[int, float] = 0,
    segmented_postfix: str = "",
    ground_truth_postfix: str = "_ground_truth",
):
    matches = match_stride_lists(
        stride_list_left=segmented_stride_list,
        stride_list_right=ground_truth,
        tolerance=tolerance,
        one_to_one=True,
        left_postfix=segmented_postfix,
        right_postfix=ground_truth_postfix,
    )
    left_index_name = segmented_stride_list.index.name + segmented_postfix
    right_index_name = ground_truth.index.name + ground_truth_postfix
    tp = matches.dropna().reset_index(drop=True)
    fp = matches[matches[right_index_name].isna()].reset_index(drop=True)
    fn = matches[matches[left_index_name].isna()].reset_index(drop=True)

    return tp, fp, fn


def match_stride_lists(
    stride_list_left: StrideList,
    stride_list_right=StrideList,
    tolerance: Union[int, float] = 0,
    one_to_one: bool = True,
    left_postfix: str = "_left",
    right_postfix: str = "_right",
) -> pd.DataFrame:
    """Find matching strides in two stride lists with a certain tolerance.

    This function will find matching strides in two stride lists as long as both start and end of a stride and its
    matching stride differ by less than the selected tolerance.
    This can be helpful to compare a the result of a segmentation to a ground truth.

    Matches will be found in both directions and mapping from the `s_id` of the left stride list to the `s_id` of the
    right stride list (and vise-versa) are returned.
    For a stride that has no valid match, it will be mapped to a NaN.
    If `one_to_one` is False, multiple matches for each stride can be found.
    This might happen, if the tolerance value is set very high or strides in the stride lists overlap.
    If `one_to_one` is True (the default) only a single match will be returned per stride.
    This will be the match with the lowest combined difference between the start and the end label.

    Parameters
    ----------
    stride_list_left
        The first stride list used for comparison
    stride_list_right : list of labels or 2D array
        The second stride list used for comparison
    tolerance
        The allowed tolerance between labels.
        Its unit depends on the units used in the stride list.
    one_to_one
        If True, only a single unique match will be returned per stride.
        If False, multiple matches are possible.
    left_postfix
        A postfix that will be append to the index name of the left stride list in the output.
    right_postfix
        A postfix that will be append to the index name of the left stride list in the output.

    Returns
    -------
    matches_df
        A 2 column dataframe with the column names `s_id{left_postfix}` and `s_id{right_postfix}`.
        Each row is a match containing the index value of the left and the right list, that belong together.
        Strides that do not have a match will be mapped to a NaN.
        The list is sorted by the index values of the left stride list.

    Examples
    --------
    TODO: Add example
    >>> ground_truth_labels = [[10,20],[21,30],[31,40],[50,60]]
    >>> predicted_labels = [[10,21],[20,34],[31,40]]
    >>> tp_labels, fp_labels, fn_labels = compare_label_lists_with_tolerance(ground_truth_labels, predicted_labels, tolerance = 2)
    >>> tp_labels
    >>> ... array([[10, 21], [31, 40]])
    >>> fp_labels
    >>> ... array([20, 34])
    >>> false_negative_labels
    >>> ... array([[21, 30], [50, 60]])

    """
    stride_list_left = set_correct_index(stride_list_left, SL_INDEX)
    stride_list_right = set_correct_index(stride_list_right, SL_INDEX)

    if not is_single_sensor_stride_list(stride_list_left):
        raise ValueError("stride_list_left is not a valid stride list")
    if not is_single_sensor_stride_list(stride_list_right):
        raise ValueError("stride_list_right is not a valid stride list")

    if left_postfix == right_postfix:
        raise ValueError("The postifix for the left and the right stride list must be different.")

    left_indices, right_indices = _match_start_end_label_lists(
        stride_list_left[["start", "end"]].to_numpy(),
        stride_list_right[["start", "end"]].to_numpy(),
        tolerance=tolerance,
        one_to_one=one_to_one,
    )

    left_index_name = stride_list_left.index.name + left_postfix
    right_index_name = stride_list_right.index.name + right_postfix

    matches_left = pd.DataFrame(index=stride_list_left.index, columns=[right_index_name])
    matches_left.index.name = left_index_name
    matches_right = pd.DataFrame(index=stride_list_right.index, columns=[left_index_name])
    matches_right.index.name = right_index_name

    stride_list_left_idx = stride_list_left.iloc[left_indices].index
    stride_list_right_idx = stride_list_right.iloc[right_indices].index

    matches_left.loc[stride_list_left_idx, right_index_name] = stride_list_right_idx
    matches_right.loc[stride_list_right_idx, left_index_name] = stride_list_left_idx
    matches_left = matches_left.reset_index()
    matches_right = matches_right.reset_index()
    matches = (
        pd.concat([matches_left, matches_right]).drop_duplicates().sort_values(left_index_name).reset_index(drop=True)
    )
    return matches


def _match_start_end_label_lists(
    list_left: np.ndarray, list_right: np.ndarray, tolerance: Union[int, float], one_to_one: bool
) -> Tuple[np.ndarray, np.ndarray]:
    distance = np.empty((len(list_left), len(list_right), 2))
    distance[..., 0] = np.subtract.outer(list_left[0], list_right[0])
    distance[..., 1] = np.subtract.outer(list_left[1], list_right[1])
    distance = np.abs(distance)
    valid_matches = distance <= tolerance
    valid_matches = valid_matches[..., 0] & valid_matches[..., 1]

    if one_to_one is True:
        left_indices = distance.sum(axis=-1).argmin(axis=0)
        left_indices = left_indices[valid_matches[left_indices, np.arange(distance.shape[1])]]
        right_indices = distance.sum(axis=-1).argmin(axis=1)
        right_indices = right_indices[valid_matches[np.arange(distance.shape[0]), right_indices]]
    else:
        left_indices, right_indices = np.where(valid_matches)
    return left_indices, right_indices
