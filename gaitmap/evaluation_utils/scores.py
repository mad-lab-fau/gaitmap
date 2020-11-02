"""A set of helper functions to score the output of the evaluation of a stride segmentation against ground truth."""

from typing import Union, Tuple, Dict, Hashable

from pandas import DataFrame

from gaitmap.utils.dataset_helper import get_multi_sensor_dataset_names


def recall_score(matches_df: Union[Dict[Hashable, DataFrame], DataFrame]) -> Union[Dict[Hashable, float], float]:
    """Compute the recall.

    The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false
    negatives. The recall is intuitively the ability of the classifier to find all the positive samples.

    The best value is 1 and the worst value is 0.

    Parameters
    ----------
    matches_df
        A 3 column dataframe with the column names `s_id{stride_list_postfix}`, `s_id{ground_truth_postfix}` and
        `match_type` or a dictionary of such dataframes.
        Each row is a match containing the index value of the left and the corresponding right one.
        The `match_type` column indicates the type of match:
        "tp" (true positive), "fp" (false positives) or "fn" (false negative) if no segmented
        counterpart exists.

    Returns
    -------
    recall_score
        This is a float, if the input is just a single dataframe or a dictionary, if the input is a dictionary of
        dataframes.

    See Also
    --------
    gaitmap.evaluation_utils.evaluate_segmented_stride_list: Generate matched_df from stride lists


    """
    is_not_dict = not isinstance(matches_df, dict)
    if is_not_dict:
        matches_df = {"__dummy__": matches_df}

    output = {}
    matches_dict = _get_match_type_dfs(matches_df)
    for sensor_name in get_multi_sensor_dataset_names(matches_dict):
        tp = len(matches_dict[sensor_name]["tp"])
        fn = len(matches_dict[sensor_name]["fn"])

        output[sensor_name] = tp / (tp + fn)

    if is_not_dict:
        output = output["__dummy__"]

    return output


def precision_score(matches_df: Union[Dict[Hashable, DataFrame], DataFrame]) -> Union[Dict[Hashable, float], float]:
    """Compute the precision.

    The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false
    positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is
    negative.

    The best value is 1 and the worst value is 0.

    Parameters
    ----------
    matches_df
        A 3 column dataframe with the column names `s_id{stride_list_postfix}`, `s_id{ground_truth_postfix}` and
        `match_type` or a dictionary of such dataframes.
        Each row is a match containing the index value of the left and the corresponding right one.
        The `match_type` column indicates the type of match:
        "tp" (true positive), "fp" (false positives) or "fn" (false negative) if no segmented
        counterpart exists.

    Returns
    -------
    precision_score
        This is a float, if the input is just a single dataframe or a dictionary, if the input is a dictionary of
        dataframes.

    See Also
    --------
    gaitmap.evaluation_utils.evaluate_segmented_stride_list: Generate matched_df from stride lists

    """
    is_not_dict = not isinstance(matches_df, dict)
    if is_not_dict:
        matches_df = {"__dummy__": matches_df}

    output = {}
    matches_dict = _get_match_type_dfs(matches_df)
    for sensor_name in get_multi_sensor_dataset_names(matches_dict):
        tp = len(matches_dict[sensor_name]["tp"])
        fp = len(matches_dict[sensor_name]["fp"])

        output[sensor_name] = tp / (tp + fp)

    if is_not_dict:
        output = output["__dummy__"]

    return output


def f1_score(matches_df: Union[Dict[Hashable, DataFrame], DataFrame]) -> Union[Dict[Hashable, float], float]:
    """Compute the F1 score, also known as balanced F-score or F-measure.

    The F1 score can be interpreted as the harmonic mean of precision and recall, where an F1 score reaches its
    best value at 1 and worst score at 0.

    The formula for the F1 score is:
    F1 = 2 * (precision * recall) / (precision + recall)

    Parameters
    ----------
    matches_df
        A 3 column dataframe with the column names `s_id{stride_list_postfix}`, `s_id{ground_truth_postfix}` and
        `match_type` or a dictionary of such dataframes.
        Each row is a match containing the index value of the left and the corresponding right one.
        The `match_type` column indicates the type of match:
        "tp" (true positive), "fp" (false positives) or "fn" (false negative) if no segmented
        counterpart exists.

    Returns
    -------
    f1_score
        This is a float, if the input is just a single dataframe or a dictionary, if the input is a dictionary of
        dataframes.

    See Also
    --------
    gaitmap.evaluation_utils.evaluate_segmented_stride_list: Generate matched_df from stride lists

    """
    is_not_dict = not isinstance(matches_df, dict)
    if is_not_dict:
        matches_df = {"__dummy__": matches_df}

    output = {}
    recall = recall_score(matches_df.copy())
    precision = precision_score(matches_df.copy())
    for sensor_name in list(recall.keys()):
        output[sensor_name] = (
            2 * (precision[sensor_name] * recall[sensor_name]) / (precision[sensor_name] + recall[sensor_name])
        )

    if is_not_dict:
        output = output["__dummy__"]

    return output


def precision_recall_f1_score(
    matches_df: Union[Dict[Hashable, DataFrame], DataFrame]
) -> Union[Dict[Hashable, Tuple[float, float, float]], Tuple[float, float, float]]:
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
        A 3 column dataframe with the column names `s_id{stride_list_postfix}`, `s_id{ground_truth_postfix}` and
        `match_type` or a dictionary of such dataframes.
        Each row is a match containing the index value of the left and the corresponding right one.
        The `match_type` column indicates the type of match:
        "tp" (true positive), "fp" (false positives) or "fn" (false negative) if no segmented
        counterpart exists.

    Returns
    -------
    score_metrics : (precision, recall, f1_score) or {"sensor_name": (precision, recall, f1_score), ...}
        Tuple of precision scores or a dictionary with the same scores.

    See Also
    --------
    gaitmap.evaluation_utils.evaluate_segmented_stride_list: Generate matched_df from stride lists

    """
    is_not_dict = not isinstance(matches_df, dict)
    if is_not_dict:
        matches_df = {"__dummy__": matches_df}

    output = {}

    precision = precision_score(matches_df.copy())
    recall = recall_score(matches_df.copy())
    f1_score_value = f1_score(matches_df.copy())

    for sensor_name in list(recall.keys()):
        output[sensor_name] = (precision[sensor_name], recall[sensor_name], f1_score_value[sensor_name])

    if is_not_dict:
        output = output["__dummy__"]

    return output


def _get_match_type_dfs(
    match_results: Union[DataFrame, Dict[Hashable, DataFrame]]
) -> Union[Dict[Hashable, Dict[str, DataFrame]], Dict[str, DataFrame]]:
    is_not_dict = not isinstance(match_results, dict)
    if is_not_dict:
        match_results = {"__dummy__": match_results}

    for dataframe_name in get_multi_sensor_dataset_names(match_results):
        matches_types = match_results[dataframe_name].groupby("match_type")
        matches_types_dict = dict()
        for group in ["tp", "fp", "fn"]:
            if group in matches_types.groups:
                matches_types_dict[group] = matches_types.get_group(group)
            else:
                matches_types_dict[group] = DataFrame(columns=match_results[dataframe_name].columns.copy())
        match_results[dataframe_name] = matches_types_dict

    if is_not_dict:
        match_results = match_results["__dummy__"]

    return match_results
