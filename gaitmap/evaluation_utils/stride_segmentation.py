"""A set of helper functions to evaluate the output of a stride segmentation against ground truth."""

from typing import Union, Tuple, Dict, Hashable

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


def recall_score(matches_df: pd.DataFrame) -> float:
    """Compute the recall.

    The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false
    negatives. The recall is intuitively the ability of the classifier to find all the positive samples.

    The best value is 1 and the worst value is 0.

    Parameters
    ----------
    matches_df
       A 3 column dataframe with the column names `s_id{segmented_postfix}`, `s_id{ground_truth_postfix}` and
       `match_type`.
       Each row is a match containing the index value of the left and the right list, that belong together.
       The `match_type` column indicates the type of match.
       For all segmented strides that have a match in the ground truth list, this will be "tp" (true positive).
       Segmented strides that do not have a match will be mapped to a NaN and the match-type will be "fp" (false
       positives)
       All ground truth strides that do not have a segmented counterpart are marked as "fn" (false negative).

    Returns
    -------
    recall score

    See Also
    --------
    gaitmap.evaluation_utils.evaluate_segmented_stride_list: Generate matched_df from stride lists


    """
    matches_dict = _get_match_type_dfs(matches_df)
    tp = len(matches_dict["tp"])
    fn = len(matches_dict["fn"])
    return tp / (tp + fn)


def precision_score(matches_df: pd.DataFrame) -> float:
    """Compute the precision.

    The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false
    positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is
    negative.

    The best value is 1 and the worst value is 0.

    Parameters
    ----------
    matches_df
       A 3 column dataframe with the column names `s_id{segmented_postfix}`, `s_id{ground_truth_postfix}` and
       `match_type`.
       Each row is a match containing the index value of the left and the right list, that belong together.
       The `match_type` column indicates the type of match.
       For all segmented strides that have a match in the ground truth list, this will be "tp" (true positive).
       Segmented strides that do not have a match will be mapped to a NaN and the match-type will be "fp" (false
       positives)
       All ground truth strides that do not have a segmented counterpart are marked as "fn" (false negative).

    Returns
    -------
    precision score

    See Also
    --------
    gaitmap.evaluation_utils.evaluate_segmented_stride_list: Generate matched_df from stride lists

    """
    matches_dict = _get_match_type_dfs(matches_df)
    tp = len(matches_dict["tp"])
    fp = len(matches_dict["fp"])
    return tp / (tp + fp)


def f1_score(matches_df: pd.DataFrame) -> float:
    """Compute the F1 score, also known as balanced F-score or F-measure.

    The F1 score can be interpreted as the harmonic mean of precision and recall, where an F1 score reaches its
    best value at 1 and worst score at 0.

    The formula for the F1 score is:
    F1 = 2 * (precision * recall) / (precision + recall)

    Parameters
    ----------
    matches_df
       A 3 column dataframe with the column names `s_id{segmented_postfix}`, `s_id{ground_truth_postfix}` and
       `match_type`.
       Each row is a match containing the index value of the left and the right list, that belong together.
       The `match_type` column indicates the type of match.
       For all segmented strides that have a match in the ground truth list, this will be "tp" (true positive).
       Segmented strides that do not have a match will be mapped to a NaN and the match-type will be "fp" (false
       positives)
       All ground truth strides that do not have a segmented counterpart are marked as "fn" (false negative).

    Returns
    -------
    F1 score

    See Also
    --------
    gaitmap.evaluation_utils.evaluate_segmented_stride_list: Generate matched_df from stride lists

    """
    recall = recall_score(matches_df)
    precision = precision_score(matches_df)
    return 2 * (precision * recall) / (precision + recall)


def precision_recall_f1_score(matches_df: pd.DataFrame) -> Tuple[float, float, float]:
    """Compute precision, recall and F1-score.

    The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false
    positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is
    negative.

    The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false
    negatives. The recall is intuitively the ability of the classifier to find all the positive samples.

    The F1 score can be interpreted as the harmonic mean of precision and recall, where an F1 score reaches its
    best value at 1 and worst score at 0.

    Parameters
    ----------
    matches_df
        A 3 column dataframe with the column names `s_id{segmented_postfix}`, `s_id{ground_truth_postfix}` and
        `match_type`.
        Each row is a match containing the index value of the left and the right list, that belong together.
        The `match_type` column indicates the type of match.
        For all segmented strides that have a match in the ground truth list, this will be "tp" (true positive).
        Segmented strides that do not have a match will be mapped to a NaN and the match-type will be "fp" (false
        positives)
        All ground truth strides that do not have a segmented counterpart are marked as "fn" (false negative).

    Returns
    -------
    precision, recall, F1-score

    See Also
    --------
    gaitmap.evaluation_utils.evaluate_segmented_stride_list: Generate matched_df from stride lists

    """
    return precision_score(matches_df), recall_score(matches_df), f1_score(matches_df)


def evaluate_segmented_stride_list(
    ground_truth: StrideList,
    segmented_stride_list: StrideList,
    tolerance: Union[int, float] = 0,
    one_to_one: bool = True,
    segmented_postfix: str = "",
    ground_truth_postfix: str = "_ground_truth",
) -> Union[pd.DataFrame, Dict[Hashable, pd.DataFrame]]:
    """Find True Positives, False Positives and True Negatives by comparing a stride list with ground truth.

    This compares a segmented stride list with a ground truth stride list and returns True Positives, False Positives
    and True Negatives matches.
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
    If ground_truth and segmented_stride_list are SingleSensorStrideLists
        A 3 column dataframe with the column names `s_id{segmented_postfix}`, `s_id{ground_truth_postfix}` and
        `match_type`.
        Each row is a match containing the index value of the left and the right list, that belong together.
        The `match_type` column indicates the type of match.
        For all segmented strides that have a match in the ground truth list, this will be "tp" (true positive).
        Segmented strides that do not have a match will be mapped to a NaN and the match-type will be "fp" (false
        positives)
        All ground truth strides that do not have a segmented counterpart are marked as "fn" (false negative).
    If ground_truth and segmented_stride_list are MultiSensorStrideLists
        A dictionary with the keys being the common sensor names and values being
        dataframes as described above.

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
    >>> matches["right_sensor"]
       s_id  s_id_ground_truth match_type
    0     0                  0         tp
    1     1                  1         tp
    2     2                  2         tp

    See Also
    --------
    match_stride_lists : Find matching strides between stride lists.

    """
    segmented_stride_list_type = is_stride_list(segmented_stride_list)
    ground_truth_type = is_stride_list(ground_truth)

    if (ground_truth_type, segmented_stride_list_type) in [("multi", "single"), ("single", "multi")]:
        raise ValidationError("The inputted lists are of not of same type")

    is_single = segmented_stride_list_type == "single" and ground_truth_type == "single"

    # if inputs are single stride lists convert them to multi stride lists with only one
    # dummy sensor so the algorithm can process them
    if is_single:
        segmented_stride_list = {"__dummy__": segmented_stride_list}
        ground_truth = {"__dummy__": ground_truth}

    matches = match_stride_lists(
        stride_list_a=segmented_stride_list,
        stride_list_b=ground_truth,
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
    tolerance: Union[int, float] = 0,
    one_to_one: bool = True,
    postfix_a: str = "_a",
    postfix_b: str = "_b",
) -> Union[pd.DataFrame, Dict[Hashable, pd.DataFrame]]:
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
        If stride_list_a and stride_list_b are SingleSensorStrideLists
            A 2 column dataframe with the column names `s_id{postfix_a}` and `s_id{postfix_b}`.
            Each row is a match containing the index value of the left and the right list, that belong together.
            Strides that do not have a match will be mapped to a NaN.
            The list is sorted by the index values of the left stride list.

        If stride_list_a and stride_list_b are MultiSensorStrideLists
            A dictionary with the keys being the common sensor names and values being
            dataframes as described above.

    Examples
    --------
    >>> stride_list_left = pd.DataFrame([[10,20],[21,30],[31,40],[50,60]], columns=["start", "end"]).rename_axis('s_id')
    >>> stride_list_right = pd.DataFrame([[10,21],[20,34],[31,40]], columns=["start", "end"]).rename_axis('s_id')
    >>> match_stride_lists(stride_list_left, stride_list_right, tolerance=2, postfix_a="_left", postfix_b="_right")
      s_id_left s_id_right
    0         0          0
    1         1        NaN
    2         2          2
    3         3        NaN
    4       NaN          1

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
    >>> test_output["right_sensor"]
      s_id_a s_id_b
    0      0      0
    1      1    NaN
    2      2      1
    3    NaN      2

    See Also
    --------
    evaluate_segmented_stride_list : Find True positive, True negatives and False positives from comparing two stride
        lists.

    """
    if postfix_a == postfix_b:
        raise ValueError("The postfix for the left and the right stride list must be different.")

    if tolerance < 0:
        raise ValueError("The tolerance must be larger 0.")

    stride_list_a_type = is_stride_list(stride_list_a)
    stride_list_b_type = is_stride_list(stride_list_b)

    if (stride_list_a_type, stride_list_b_type) in [("multi", "single"), ("single", "multi")]:
        raise ValidationError("The inputted lists are of not of same type")

    matches = {}

    if stride_list_a_type == "single" and stride_list_b_type == "single":

        matches = _match_single_stride_lists(
            stride_list_a,
            stride_list_b,
            tolerance=tolerance,
            one_to_one=one_to_one,
            postfix_a=postfix_a,
            postfix_b=postfix_b,
        )

    if stride_list_a_type == "multi" and stride_list_b_type == "multi":
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

        for sensor_name in sensor_names_list:
            matches[sensor_name] = _match_single_stride_lists(
                stride_list_a[sensor_name],
                stride_list_b[sensor_name],
                tolerance=tolerance,
                one_to_one=one_to_one,
                postfix_a=postfix_a,
                postfix_b=postfix_b,
            )

    return matches


def _match_single_stride_lists(
    stride_list_a: StrideList,
    stride_list_b: StrideList,
    tolerance: Union[int, float] = 0,
    one_to_one: bool = True,
    postfix_a: str = "_a",
    postfix_b: str = "_b",
) -> pd.DataFrame:
    stride_list_a = set_correct_index(stride_list_a, SL_INDEX)
    stride_list_b = set_correct_index(stride_list_b, SL_INDEX)

    left_indices, right_indices = _match_start_end_label_lists(
        stride_list_a[["start", "end"]].to_numpy(),
        stride_list_b[["start", "end"]].to_numpy(),
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


def _match_start_end_label_lists(
    list_left: np.ndarray, list_right: np.ndarray, tolerance: Union[int, float], one_to_one: bool
) -> Tuple[np.ndarray, np.ndarray]:
    if len(list_left) == 0 or len(list_right) == 0:
        return np.array([]), np.array([])
    distance = np.empty((len(list_left), len(list_right), 2))
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


def _get_match_type_dfs(
    match_results: Union[pd.DataFrame, Dict[Hashable, pd.DataFrame]]
) -> Union[Dict[Hashable, Dict[str, pd.DataFrame]], Dict[str, pd.DataFrame]]:
    is_dict = isinstance(match_results, dict)
    if not is_dict:
        match_results = {"__dummy__": match_results}

    for dataframe_name in get_multi_sensor_dataset_names(match_results):
        matches_types = match_results[dataframe_name].groupby("match_type")
        matches_types_dict = dict()
        for group in ["tp", "fp", "fn"]:
            if group in matches_types.groups:
                matches_types_dict[group] = matches_types.get_group(group)
            else:
                matches_types_dict[group] = pd.DataFrame(columns=match_results[dataframe_name].columns.copy())
        match_results[dataframe_name] = matches_types_dict

    if not is_dict:
        match_results = match_results["__dummy__"]

    return match_results
