"""A set of helper functions to evaluate the output of a stride segmentation against ground truth."""

from typing import Union, Tuple, Dict, Hashable, List

import numpy as np
import pandas as pd

from gaitmap.utils.consts import SL_INDEX
from gaitmap.utils.dataset_helper import (
    StrideList,
    set_correct_index,
    get_multi_sensor_dataset_names,
    is_stride_list,
)
from gaitmap.utils.exceptions import ValidationError


def evaluate_segmented_stride_list(
    ground_truth: StrideList,
    segmented_stride_list: StrideList,
    tolerance: Union[int, float] = 0,
    one_to_one: bool = True,
    segmented_postfix: str = "",
    ground_truth_postfix: str = "_ground_truth",
) -> Union[pd.DataFrame, Dict[Hashable, pd.DataFrame]]:
    """Find True Positives, False Positives and True Negatives by comparing a segmented stride list with ground truth.

    This compares a segmented stride list with a ground truth segmented stride list and returns True Positives,
    False Positives and True Negatives matches.
    The comparison is purely based on the start and end values of each stride in the lists.
    Two strides are considered a positive match, if both their start and their end values differ by less than the
    threshold.
    If multiple strides of the segmented stride list would match to the ground truth (or vise-versa) only the stride
    with the lowest combined distance is considered.
    This might still lead to unexpected results in certain cases.
    It is highly recommended to order the stride lists and remove strides with large overlaps before applying this
    method to get reliable results.

    Parameters
    ----------
    ground_truth
        The ground truth stride list.
    segmented_stride_list
        The list of segmented strides.
    tolerance
        The allowed tolerance between labels.
        Its unit depends on the units used in the stride lists.
    one_to_one
        If True, only a single unique match will be returned per stride.
        If False, multiple matches are possible.
        If this is set to False, some calculated metrics from these matches might not be well defined!
    segmented_postfix
        A postfix that will be append to the index name of the segmented stride list in the output.
    ground_truth_postfix
        A postfix that will be append to the index name of the ground truth in the output.

    Returns
    -------
    matches
        A 3 column dataframe with the column names `s_id{segmented_postfix}`, `s_id{ground_truth_postfix}` and
        `match_type`.
        Each row is a match containing the index value of the left and the right list, that belong together.
        The `match_type` column indicates the type of match.
        For all segmented strides that have a match in the ground truth list, this will be "tp" (true positive).
        Segmented strides that do not have a match will be mapped to a NaN and the match-type will be "fp" (false
        positives)
        All ground truth strides that do not have a segmented counterpart are marked as "fn" (false negative).
        In case MultiSensorStrideLists were used as inputs, a dictionary of such values are returned.


    Examples
    --------
    >>> stride_list_ground_truth = pd.DataFrame([[10,21],[20,34],[31,40]], columns=["start", "end"]).rename_axis('s_id')
    >>> stride_list_seg = pd.DataFrame([[10,20],[21,30],[31,40],[50,60]], columns=["start", "end"]).rename_axis('s_id')
    >>> matches = evaluate_segmented_stride_list(stride_list_ground_truth, stride_list_seg, tolerance=2)
    >>> matches
      s_id s_id_ground_truth match_type
    0    0                 0         tp
    1    1               NaN         fp
    2    2                 2         tp
    3    3               NaN         fp
    4  NaN                 1         fn

    >>> stride_list_ground_truth_left = pd.DataFrame(
    ...     [[10,21],[20,34],[31,40]],
    ...     columns=["start", "end"]
    ... ).rename_axis('s_id')
    >>> stride_list_ground_truth_right = pd.DataFrame(
    ...     [[10,21],[20,34],[31,40]],
    ...     columns=["start", "end"]
    ... ).rename_axis('s_id')
    ...
    >>> stride_list_seg_left = pd.DataFrame(
    ...     [[10,20],[21,30],[31,40],[50,60]],
    ...     columns=["start", "end"]
    ... ).rename_axis('s_id')
    >>> stride_list_seg_right = pd.DataFrame([[10,21],[20,34],[31,40]], columns=["start", "end"]).rename_axis('s_id')
    ...
    >>> matches = evaluate_segmented_stride_list(
    ...     {"left_sensor": stride_list_ground_truth_left, "right_sensor": stride_list_ground_truth_right},
    ...     {"left_sensor": stride_list_seg_left, "right_sensor": stride_list_seg_right},
    ...     tolerance=2
    ... )
    >>> matches["left_sensor"]
      s_id s_id_ground_truth match_type
    0    0                 0         tp
    1    1               NaN         fp
    2    2                 2         tp
    3    3               NaN         fp
    4  NaN                 1         fn

    See Also
    --------
    match_stride_lists : Find matching strides between stride lists.

    """
    return _evaluate_stride_list(
        ground_truth,
        segmented_stride_list,
        tolerance=tolerance,
        one_to_one=one_to_one,
        segmented_postfix=segmented_postfix,
        ground_truth_postfix=ground_truth_postfix,
    )


def _evaluate_stride_list(
    ground_truth: StrideList,
    segmented_stride_list: StrideList,
    match_cols: Union[str, Tuple[str, str]] = ("start", "end"),
    tolerance: Union[int, float] = 0,
    one_to_one: bool = True,
    segmented_postfix: str = "",
    ground_truth_postfix: str = "_ground_truth",
) -> Union[pd.DataFrame, Dict[Hashable, pd.DataFrame]]:
    segmented_stride_list_type = is_stride_list(segmented_stride_list)
    ground_truth_type = is_stride_list(ground_truth)

    if ground_truth_type != segmented_stride_list_type:
        raise ValidationError("The inputted lists are of not of same type")

    is_single = segmented_stride_list_type == "single"

    # if inputs are single stride lists convert them to multi stride lists with only one
    # dummy sensor so the algorithm can process them
    if is_single:
        segmented_stride_list = {"__dummy__": segmented_stride_list}
        ground_truth = {"__dummy__": ground_truth}

    match_cols = [match_cols] if isinstance(match_cols, str) else list(match_cols)

    matches = _match_stride_lists(
        stride_list_a=segmented_stride_list,
        stride_list_b=ground_truth,
        match_cols=match_cols,
        tolerance=tolerance,
        one_to_one=one_to_one,
        postfix_a=segmented_postfix,
        postfix_b=ground_truth_postfix,
    )

    for sensor_name in get_multi_sensor_dataset_names(matches):
        segmented_index_name = segmented_stride_list[sensor_name].index.name + segmented_postfix
        ground_truth_index_name = ground_truth[sensor_name].index.name + ground_truth_postfix
        matches[sensor_name].loc[~matches[sensor_name].isna().any(axis=1), "match_type"] = "tp"
        matches[sensor_name].loc[matches[sensor_name][ground_truth_index_name].isna(), "match_type"] = "fp"
        matches[sensor_name].loc[matches[sensor_name][segmented_index_name].isna(), "match_type"] = "fn"

    if is_single:
        matches = matches["__dummy__"]

    return matches


def match_stride_lists(
    stride_list_a: StrideList,
    stride_list_b: StrideList,
    match_cols: Union[str, List[str]] = None,
    tolerance: Union[int, float] = 0,
    one_to_one: bool = True,
    postfix_a: str = "_a",
    postfix_b: str = "_b",
) -> Union[pd.DataFrame, Dict[Hashable, pd.DataFrame]]:
    """Find matching strides in two stride lists with a certain tolerance.

    This function will find matching strides in two stride lists as long as both start and end of a stride and its
    matching stride differ by less than the selected tolerance.
    This can be helpful to compare a the result of a segmentation to a ground truth.
    In case both stride lists are multi-sensor stride lists, matching will be performed between all common sensors of
    the stride lists.
    Additional sensors are simply ignored,

    Matches will be found in both directions and mapping from the `s_id` of the left stride list to the `s_id` of the
    right stride list (and vise-versa) are returned.
    For a stride that has no valid match, it will be mapped to a NaN.
    If `one_to_one` is False, multiple matches for each stride can be found.
    This might happen, if the tolerance value is set very high or strides in the stride lists overlap.
    If `one_to_one` is True (the default) only a single match will be returned per stride.
    This will be the match with the lowest combined difference between the start and the end label.
    In case multiple strides have the same combined difference, the one that occurs first in the list is chosen.
    This might still lead to unexpected results in certain cases.
    It is highly recommended to order the stride lists and remove strides with large overlaps before applying this
    method to get reliable results.

    Parameters
    ----------
    stride_list_a
        The first stride list used for comparison
    stride_list_b
        The second stride list used for comparison
    match_cols
        A string or a list of strings that describes what you want to match.
        Default is ["start", "end"].
    tolerance
        The allowed tolerance between labels.
        Its unit depends on the units used in the stride lists.
    one_to_one
        If True, only a single unique match will be returned per stride.
        If False, multiple matches are possible.
    postfix_a
        A postfix that will be append to the index name of the left stride list in the output.
    postfix_b
        A postfix that will be append to the index name of the left stride list in the output.

    Returns
    -------
    matches
        A 2 column dataframe with the column names `s_id{postfix_a}` and `s_id{postfix_b}`.
        Each row is a match containing the index value of the left and the right list, that belong together.
        Strides that do not have a match will be mapped to a NaN.
        The list is sorted by the index values of the left stride list.
        In case MultiSensorStrideLists were used as inputs, a dictionary of such values are returned.

    Examples
    --------
    Single Sensor:

    >>> stride_list_left = pd.DataFrame([[10,20],[21,30],[31,40],[50,60]], columns=["start", "end"]).rename_axis('s_id')
    >>> stride_list_right = pd.DataFrame([[10,21],[20,34],[31,40]], columns=["start", "end"]).rename_axis('s_id')
    >>> match_stride_lists(stride_list_left, stride_list_right, tolerance=2, postfix_a="_left", postfix_b="_right")
      s_id_left s_id_right
    0         0          0
    1         1        NaN
    2         2          2
    3         3        NaN
    4       NaN          1

    Multi Sensor:

    >>> stride_list_left_11 = pd.DataFrame(
    ...     [[10,20],[21,30],[31,40],[50,60]],
    ...     columns=["start", "end"]
    ... ).rename_axis('s_id')
    >>> stride_list_right_12 = pd.DataFrame([[10,21],[20,34],[31,40]], columns=["start", "end"]).rename_axis('s_id')
    ...
    >>> stride_list_left_21 = pd.DataFrame(
    ...     [[10,20],[31,41],[21,31],[50,60]],
    ...     columns=["start", "end"]
    ... ).rename_axis('s_id')
    >>> stride_list_right_22 = pd.DataFrame([[10,22],[31, 41],[20, 36]], columns=["start", "end"]).rename_axis('s_id')
    ...
    >>> test_output = match_stride_lists(
    ...     {"left_sensor": stride_list_left_11, "right_sensor": stride_list_right_12},
    ...     {"left_sensor": stride_list_left_21, "right_sensor": stride_list_right_22},
    ...     tolerance=1
    ... )
    >>> test_output["left_sensor"]
       s_id_a  s_id_b
    0       0       0
    1       1       2
    2       2       1
    3       3       3

    See Also
    --------
    evaluate_segmented_stride_list : Find True positive, True negatives and False positives from comparing two stride
        lists.

    """
    if not match_cols:
        match_cols = ["start", "end"]
    elif isinstance(match_cols, str):
        match_cols = [match_cols]

    return _match_stride_lists(
        stride_list_a,
        stride_list_b,
        match_cols=match_cols,
        tolerance=tolerance,
        one_to_one=one_to_one,
        postfix_a=postfix_a,
        postfix_b=postfix_b,
    )


def _match_stride_lists(
    stride_list_a: StrideList,
    stride_list_b: StrideList,
    match_cols: List[str],
    tolerance: Union[int, float] = 0,
    one_to_one: bool = True,
    postfix_a: str = "_a",
    postfix_b: str = "_b",
) -> Union[pd.DataFrame, Dict[Hashable, pd.DataFrame]]:
    if postfix_a == postfix_b:
        raise ValueError("The postfix for the left and the right stride list must be different.")

    if tolerance < 0:
        raise ValueError("The tolerance must be larger 0.")

    stride_list_a_type = is_stride_list(stride_list_a)
    stride_list_b_type = is_stride_list(stride_list_b)

    if stride_list_a_type != stride_list_b_type:
        raise ValidationError("The inputted lists are of not of same type")

    matches = {}

    if stride_list_a_type == "single":
        if not set(match_cols).issubset(stride_list_a.columns):
            raise ValueError("The columns used to match are not in inputted stride lists")

        matches = _match_single_stride_lists(
            stride_list_a,
            stride_list_b,
            match_cols=match_cols,
            tolerance=tolerance,
            one_to_one=one_to_one,
            postfix_a=postfix_a,
            postfix_b=postfix_b,
        )

    if stride_list_a_type == "multi":

        # get sensor names that are in stride_list_a AND in stride_list_b
        sensor_names_list = sorted(
            list(
                set(get_multi_sensor_dataset_names(stride_list_a)).intersection(
                    get_multi_sensor_dataset_names(stride_list_b)
                )
            )
        )

        if not sensor_names_list:
            raise ValidationError("The passed MultiSensorStrideLists do not have any common sensors.")

        if not set(match_cols).issubset(stride_list_a[sensor_names_list[0]].columns):
            raise ValueError("The columns used to match are not in inputted stride lists")

        for sensor_name in sensor_names_list:

            matches[sensor_name] = _match_single_stride_lists(
                stride_list_a[sensor_name],
                stride_list_b[sensor_name],
                match_cols=match_cols,
                tolerance=tolerance,
                one_to_one=one_to_one,
                postfix_a=postfix_a,
                postfix_b=postfix_b,
            )

    return matches


def _match_single_stride_lists(
    stride_list_a: StrideList,
    stride_list_b: StrideList,
    match_cols: List[str],
    tolerance: Union[int, float] = 0,
    one_to_one: bool = True,
    postfix_a: str = "_a",
    postfix_b: str = "_b",
) -> pd.DataFrame:
    stride_list_a = set_correct_index(stride_list_a, SL_INDEX)
    stride_list_b = set_correct_index(stride_list_b, SL_INDEX)

    left_indices, right_indices = _match_label_lists(
        stride_list_a[match_cols].to_numpy(),
        stride_list_b[match_cols].to_numpy(),
        tolerance=tolerance,
        one_to_one=one_to_one,
    )

    left_index_name = stride_list_a.index.name + postfix_a
    right_index_name = stride_list_b.index.name + postfix_b

    matches_left = pd.DataFrame(index=stride_list_a.index.copy(), columns=[right_index_name])
    matches_left.index.name = left_index_name

    matches_right = pd.DataFrame(index=stride_list_b.index.copy(), columns=[left_index_name])
    matches_right.index.name = right_index_name

    stride_list_left_idx = stride_list_a.iloc[left_indices].index
    stride_list_right_idx = stride_list_b.iloc[right_indices].index

    matches_left.loc[stride_list_left_idx, right_index_name] = stride_list_right_idx
    matches_right.loc[stride_list_right_idx, left_index_name] = stride_list_left_idx

    matches_left = matches_left.reset_index()
    matches_right = matches_right.reset_index()

    matches = (
        pd.concat([matches_left, matches_right])
        .drop_duplicates()
        .sort_values([left_index_name, right_index_name])
        .reset_index(drop=True)
    )

    return matches


def _match_label_lists(
    list_left: np.ndarray, list_right: np.ndarray, tolerance: Union[int, float], one_to_one: bool
) -> Tuple[np.ndarray, np.ndarray]:
    if len(list_left) == 0 or len(list_right) == 0:
        return np.array([]), np.array([])

    nr_of_cols_left = list_left.shape[1]
    nr_of_cols_right = list_left.shape[1]

    if nr_of_cols_left > 2 or nr_of_cols_right > 2:
        raise ValidationError("Can not compare more than 2 columns at a time")

    both_1d = nr_of_cols_left == 1 and nr_of_cols_right == 1

    distance = np.empty((len(list_left), len(list_right), 2))

    if both_1d:
        list_left = np.array([[x[0], x[0]] for x in list_left])
        list_right = np.array([[x[0], x[0]] for x in list_right])

    distance[..., 0] = np.subtract.outer(list_left[:, 0], list_right[:, 0])
    distance[..., 1] = np.subtract.outer(list_left[:, 1], list_right[:, 1])

    distance = np.abs(distance)
    valid_matches = distance <= tolerance

    valid_matches = valid_matches[..., 0] & valid_matches[..., 1]

    if one_to_one is True:
        argmin_array_left = np.zeros(valid_matches.shape).astype(bool)
        left_indices = distance.sum(axis=-1).argmin(axis=0)
        argmin_array_left[left_indices, np.arange(distance.shape[1])] = True
        right_indices = distance.sum(axis=-1).argmin(axis=1)
        argmin_array_right = np.zeros(valid_matches.shape).astype(bool)
        argmin_array_right[np.arange(distance.shape[0]), right_indices] = True
        valid_matches &= argmin_array_left & argmin_array_right
    left_indices, right_indices = np.where(valid_matches)

    return left_indices, right_indices
