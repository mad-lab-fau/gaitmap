"""A set of helper functions to evaluate the output of an event stride segmentation against ground truth."""

from typing import Dict, Union

from pandas import DataFrame
from typing_extensions import Literal

from gaitmap.evaluation_utils.stride_segmentation import _evaluate_stride_list
from gaitmap.utils._types import _Hashable
from gaitmap.utils.datatype_helper import StrideList


def evaluate_stride_event_list(
    *,
    ground_truth: StrideList,
    stride_event_list: StrideList,
    match_cols: Literal["pre_ic", "ic", "min_vel", "tc"],
    tolerance: Union[int, float] = 0,
    one_to_one: bool = True,
    stride_list_postfix: str = "",
    ground_truth_postfix: str = "_ground_truth",
) -> Union[DataFrame, Dict[_Hashable, DataFrame]]:
    """Find True Positives, False Positives and True Negatives by comparing an stride event list with ground truth.

    This compares a stride event list with a ground truth stride event list and returns True Positives,
    False Positives and True Negatives matches.
    The comparison is based on the chosen column ("pre_ic", "ic", "min_vel", or "tc").
    Two strides are considered a positive match, if the selected event differs by less than the
    threshold.

    By default (controlled by the one-to-one parameter), if multiple strides of the stride event list would match to
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
        The ground truth stride event list.
    stride_event_list
        The stride event list.
    match_cols
        A string that describes what you want to match.
        Must be one of pre_ic, ic, min_vel or tc.
    tolerance
        The allowed tolerance between labels.
        Its unit depends on the units used in the stride lists.
    one_to_one
        If True, only a single unique match per stride is considered.
        If False, multiple matches are possible.
        If this is set to False, some calculated metrics from these matches might not be well defined!
    stride_list_postfix
        A postfix that will be append to the index name of the stride event list in the output.
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
        All ground truth strides that do not have a counterpart are marked as "fn" (false negative).
        In case MultiSensorStrideLists were used as inputs, a dictionary of such dataframes is returned.


    Examples
    --------
    >>> stride_list_ground_truth = DataFrame(
    ...     [[10,21, 10],[20,34, 30],[31,40, 20]],
    ...     columns=["start", "end", "ic"]
    ... ).rename_axis('s_id')
    >>> stride_list_seg = DataFrame(
    ...     [[10,20, 10],[21,30, 30],[31,40, 22]],
    ...     columns=["start", "end", "ic"]
    ... ).rename_axis('s_id')
    >>> matches = evaluate_stride_event_list(
    ...     ground_truth=stride_list_ground_truth,
    ...     stride_event_list=stride_list_seg,
    ...     match_cols="ic",
    ...     tolerance=3
    ... )
    >>> matches
       s_id  s_id_ground_truth match_type
    0     0                  0         tp
    1     1                  1         tp
    2     2                  2         tp

    >>> stride_list_ground_truth_left = DataFrame(
    ...     [[10,21,30],[20,34,20],[31,40,10], [10, 30 ,60]],
    ...     columns=["start", "end", "ic"]
    ... ).rename_axis('s_id')
    >>> stride_list_ground_truth_right = DataFrame(
    ...     [[10,21,1],[20,34,2],[31,40,3]],
    ...     columns=["start", "end", "ic"]
    ... ).rename_axis('s_id')
    ...
    >>> stride_list_seg_left = DataFrame(
    ...     [[10,20, 30],[21,30,20],[31,40,13]],
    ...     columns=["start", "end", "ic"]
    ... ).rename_axis('s_id')
    >>> stride_list_seg_right = DataFrame(
    ...     [[10,21, 1],[20,34, 2],[31,40, 3]],
    ...     columns=["start", "end", "ic"]
    ... ).rename_axis('s_id')
    ...
    >>> matches_multi = evaluate_stride_event_list(
    ...     ground_truth={"left_sensor": stride_list_ground_truth_left, "right_sensor": stride_list_ground_truth_right},
    ...     stride_event_list={"left_sensor": stride_list_seg_left, "right_sensor": stride_list_seg_right},
    ...     match_cols="ic",
    ...     tolerance=2
    ... )
    >>> matches_multi["left_sensor"]
      s_id s_id_ground_truth match_type
    0    0                 0         tp
    1    1                 1         tp
    2    2               NaN         fp
    3  NaN                 2         fn
    4  NaN                 3         fn

    See Also
    --------
    gaitmap.evaluation_utils.match_stride_lists: Find matching strides between stride lists.
    gaitmap.evaluation_utils.evaluate_segmented_stride_list: Find matching strides between segmented stride lists.

    """
    # We do not check here if a valid event col is passed.
    # We later check if the selected column is in all the dataframes.
    # This is actually all we care about, even, if the user selects an event, we would not expect.
    return _evaluate_stride_list(
        ground_truth,
        stride_event_list,
        match_cols=match_cols,
        tolerance=tolerance,
        one_to_one=one_to_one,
        stride_list_postfix=stride_list_postfix,
        ground_truth_postfix=ground_truth_postfix,
    )
