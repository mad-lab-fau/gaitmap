"""A set of helper functions to evaluate the output of a stride segmentation against ground truth."""

from typing import Union, Tuple, Dict, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import minkowski_distance
from scipy.spatial.ckdtree import cKDTree

from gaitmap.utils._types import _Hashable
from gaitmap.utils.consts import SL_INDEX
from gaitmap.utils.datatype_helper import (
    StrideList,
    set_correct_index,
    get_multi_sensor_names,
    is_stride_list,
    SingleSensorStrideList,
)
from gaitmap.utils.exceptions import ValidationError


def evaluate_segmented_stride_list(
    *,
    ground_truth: StrideList,
    segmented_stride_list: StrideList,
    tolerance: Union[int, float] = 0,
    one_to_one: bool = True,
    stride_list_postfix: str = "",
    ground_truth_postfix: str = "_ground_truth",
) -> Union[pd.DataFrame, Dict[_Hashable, pd.DataFrame]]:
    """Find True Positives, False Positives and True Negatives by comparing a segmented stride list with ground truth.

    This compares a segmented stride list with a ground truth segmented stride list and returns True Positives,
    False Positives and True Negatives matches.
    The comparison is purely based on the start and end values of each stride in the lists.
    Two strides are considered a positive match, if both their start and their end values differ by less than the
    threshold.

    By default (controlled by the one-to-one parameter), if multiple strides of the segmented stride list would match to
    a single ground truth stride (or vise-versa), only the stride with the lowest distance is considered an actual
    match.
    If `one_to_one` is set to False, all matches would be considered True positives.
    This might lead to unexpected results in certain cases and should not be used to calculate traditional metrics like
    precision and recall.

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
        If True, only a single unique match per stride is considered.
        If False, multiple matches are possible.
        If this is set to False, some calculated metrics from these matches might not be well defined!
    stride_list_postfix
        A postfix that will be append to the index name of the segmented stride list in the output.
    ground_truth_postfix
        A postfix that will be append to the index name of the ground truth in the output.

    Returns
    -------
    matches
        A 3 column dataframe with the column names `s_id{stride_list_postfix}`, `s_id{ground_truth_postfix}` and
        `match_type`.
        Each row is a match containing the index value of the left and the right list, that belong together.
        The `match_type` column indicates the type of match.
        For all segmented strides that have a match in the ground truth list, this will be "tp" (true positive).
        Segmented strides that do not have a match will be mapped to a NaN and the match-type will be "fp" (false
        positives)
        All ground truth strides that do not have a segmented counterpart are marked as "fn" (false negative).
        In case MultiSensorStrideLists were used as inputs, a dictionary of such dataframes is returned.


    Examples
    --------
    >>> stride_list_ground_truth = pd.DataFrame([[10,21],[20,34],[31,40]], columns=["start", "end"]).rename_axis('s_id')
    >>> stride_list_seg = pd.DataFrame([[10,20],[21,30],[31,40],[50,60]], columns=["start", "end"]).rename_axis('s_id')
    >>> matches = evaluate_segmented_stride_list(
    ...     ground_truth=stride_list_ground_truth,
    ...     segmented_stride_list=stride_list_seg,
    ...     tolerance=2
    ... )
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
    ...     ground_truth={"left_sensor": stride_list_ground_truth_left, "right_sensor": stride_list_ground_truth_right},
    ...     segmented_stride_list={"left_sensor": stride_list_seg_left, "right_sensor": stride_list_seg_right},
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
    gaitmap.evaluation_utils.match_stride_lists: Find matching strides between stride lists.
    gaitmap.evaluation_utils.evaluate_stride_event_list: Find matching strides between stride event lists.

    """
    return _evaluate_stride_list(
        ground_truth,
        segmented_stride_list,
        tolerance=tolerance,
        one_to_one=one_to_one,
        stride_list_postfix=stride_list_postfix,
        ground_truth_postfix=ground_truth_postfix,
    )


def _evaluate_stride_list(
    ground_truth: StrideList,
    segmented_stride_list: StrideList,
    match_cols: Union[str, Sequence[str]] = ("start", "end"),
    tolerance: Union[int, float] = 0,
    one_to_one: bool = True,
    stride_list_postfix: str = "",
    ground_truth_postfix: str = "_ground_truth",
) -> Union[pd.DataFrame, Dict[_Hashable, pd.DataFrame]]:
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

    matches = _match_stride_lists(
        stride_list_a=segmented_stride_list,
        stride_list_b=ground_truth,
        match_cols=match_cols,
        tolerance=tolerance,
        one_to_one=one_to_one,
        postfix_a=stride_list_postfix,
        postfix_b=ground_truth_postfix,
    )

    for sensor_name in get_multi_sensor_names(matches):
        segmented_index_name = segmented_stride_list[sensor_name].index.name + stride_list_postfix
        ground_truth_index_name = ground_truth[sensor_name].index.name + ground_truth_postfix
        matches[sensor_name].loc[~matches[sensor_name].isna().any(axis=1), "match_type"] = "tp"
        matches[sensor_name].loc[matches[sensor_name][ground_truth_index_name].isna(), "match_type"] = "fp"
        matches[sensor_name].loc[matches[sensor_name][segmented_index_name].isna(), "match_type"] = "fn"

    if is_single:
        matches = matches["__dummy__"]

    return matches


def match_stride_lists(
    *,
    stride_list_a: StrideList,
    stride_list_b: StrideList,
    match_cols: Union[str, Sequence[str]] = ("start", "end"),
    tolerance: Union[int, float] = 0,
    one_to_one: bool = True,
    postfix_a: str = "_a",
    postfix_b: str = "_b",
) -> Union[pd.DataFrame, Dict[_Hashable, pd.DataFrame]]:
    """Find matching strides in two stride lists with a certain tolerance.

    This function will find matching strides in two stride lists as long as all selected columns/event of a stride
    and its matching stride differ by less than the selected tolerance.
    This can be helpful to compare the result of a segmentation or event detection to a ground truth.
    In case both stride lists are multi-sensor stride lists, matching will be performed between all common sensors of
    the stride lists.
    Additional sensors are simply ignored.

    Matches will be found in both directions and mapping from the `s_id` of the left stride list to the `s_id` of the
    right stride list (and vise-versa) are returned.
    For a stride that has no valid match, it will be mapped to a NaN.
    If `one_to_one` is False, multiple matches for each stride can be found.
    This might happen, if the tolerance value is set very high or strides in the stride lists overlap.
    If `one_to_one` is True (the default) only a single match will be returned per stride.
    This will be the match with the lowest combined difference over all the selected columns/events.
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
    >>> match_stride_lists(
    ...     stride_list_a=stride_list_left,
    ...     stride_list_b=stride_list_right,
    ...     tolerance=2,
    ...     postfix_a="_left",
    ...     postfix_b="_right"
    ... )
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
    ...     stride_list_a={"left_sensor": stride_list_left_11, "right_sensor": stride_list_right_12},
    ...     stride_list_b={"left_sensor": stride_list_left_21, "right_sensor": stride_list_right_22},
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
    match_cols: Union[str, Sequence[str]],
    tolerance: Union[int, float] = 0,
    one_to_one: bool = True,
    postfix_a: str = "_a",
    postfix_b: str = "_b",
) -> Union[pd.DataFrame, Dict[_Hashable, pd.DataFrame]]:
    if postfix_a == postfix_b:
        raise ValueError("The postfix for the left and the right stride list must be different.")

    if tolerance < 0:
        raise ValueError("The tolerance must be larger 0.")

    match_cols = [match_cols] if isinstance(match_cols, str) else list(match_cols)

    stride_list_a_type = is_stride_list(stride_list_a)
    stride_list_b_type = is_stride_list(stride_list_b)

    if stride_list_a_type != stride_list_b_type:
        raise ValidationError("The inputted lists are of not of same type")

    matches = {}

    if stride_list_a_type == "single":
        matches = _match_single_stride_lists(
            stride_list_a,
            stride_list_b,
            match_cols=match_cols,
            tolerance=tolerance,
            one_to_one=one_to_one,
            postfix_a=postfix_a,
            postfix_b=postfix_b,
        )
    else:
        # get sensor names that are in stride_list_a AND in stride_list_b
        sensor_names_list = sorted(
            list(set(get_multi_sensor_names(stride_list_a)).intersection(get_multi_sensor_names(stride_list_b)))
        )
        if not sensor_names_list:
            raise ValidationError("The passed MultiSensorStrideLists do not have any common sensors.")
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
    stride_list_a: SingleSensorStrideList,
    stride_list_b: SingleSensorStrideList,
    match_cols: Union[str, Sequence[str]],
    tolerance: Union[int, float] = 0,
    one_to_one: bool = True,
    postfix_a: str = "_a",
    postfix_b: str = "_b",
) -> pd.DataFrame:
    if not (set(match_cols).issubset(stride_list_a.columns) and set(match_cols).issubset(stride_list_b.columns)):
        raise ValueError(
            "One or more selected columns ({}) are missing in at least one of the provided stride lists".format(
                match_cols
            )
        )
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
    """Find matches in two lists based on the distance between their vectors.

    Parameters
    ----------
    list_left : array with shape (n, d)
        An n long array of d-dimensional vectors
    list_right : array with shape (m, d)
        An m long array of d-dimensional vectors
    tolerance
        Max allowed Chebyshev distance between matches
    one_to_one
        If True only valid one-to-one matches are returned (see more below)

    Returns
    -------
    left_indices
        Indices from the left list that have a match in the right list.
        If `one_to_one` is False, indices might repeat.
    right_indices
        Indices from the right list that have a match in the left list.
        If `one_to_one` is False, indices might repeat.
        A valid match pare is then `(left_indices[i], right_indices[i]) for all i.

    Notes
    -----
    This function supports 2 modes:

    `one_to_one` = False:
        In this mode every match is returned as long the distance in all dimensions between the matches is at most
        tolerance.
        This is equivalent to the Chebyshev distance between the matches
        (aka `np.max(np.abs(left_match - right_match)) < tolerance`).
        This means multiple matches for each vector will be returned.
        This means the respective indices will occur multiple times in the output vectors.
    `one_to_one` = True:
        In this mode only a single match per index is allowed in both directions.
        This means that every index will only occur once in the output arrays.
        If multiple matches are possible based on the tolerance of the Chebyshev distance, the closest match will be
        selected based on the Manhatten distance (aka `np.sum(np.abs(left_match - right_match`).
        Only this match will be returned.
        Note, that in the implementation, we first get the closest match based on the Manhatten distance and check in a
        second step if this closed match is also valid based on the Chebyshev distance.

    """
    if len(list_left) == 0 or len(list_right) == 0:
        return np.array([]), np.array([])

    right_tree = cKDTree(list_right)
    left_tree = cKDTree(list_left)

    if one_to_one is False:
        # p = np.inf is used to select the Chebyshev distance
        keys = list(zip(*right_tree.sparse_distance_matrix(left_tree, tolerance, p=np.inf).keys()))
        # All values are returned that have a valid match
        return (np.array([]), np.array([])) if len(keys) == 0 else (np.array(keys[1]), np.array(keys[0]))

    # one_to_one is True
    # We calculate the closest neighbor based on the Manhatten distance in both directions and then find only the cases
    # were the right side closest neighbor resulted in the same pairing as the left side closest neighbor ensuring
    # that we have true one-to-one-matches

    # p = 1 is used to select the Manhatten distance
    l_nearest_distance, l_nearest_neighbor = right_tree.query(list_left, p=1, workers=-1)
    _, r_nearest_neighbor = left_tree.query(list_right, p=1, workers=-1)

    # Filter the once that are true one-to-one matches
    l_indices = np.arange(len(list_left))
    combined_indices = np.vstack([l_indices, l_nearest_neighbor]).T
    boolean_map = r_nearest_neighbor[l_nearest_neighbor] == l_indices
    valid_matches = combined_indices[boolean_map]

    # Check if the remaining matches are inside our Chebyshev tolerance distance.
    # If not, delete them.
    valid_matches_distance = l_nearest_distance[boolean_map]
    index_large_matches = np.where(valid_matches_distance > tolerance)[0]
    if index_large_matches.size > 0:
        # Minkowski with p = np.inf uses the Chebyshev distance
        output = (
            minkowski_distance(
                list_left[index_large_matches], list_right[valid_matches[index_large_matches, 1]], p=np.inf
            )
            > tolerance
        )

        valid_matches = np.delete(valid_matches, index_large_matches[output], axis=0)

    valid_matches = valid_matches.T

    return valid_matches[0], valid_matches[1]
