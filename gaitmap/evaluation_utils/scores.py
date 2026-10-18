"""A set of helper functions to score the output of the evaluation of a stride segmentation against ground truth."""

import warnings
from typing import Literal, TypedDict, overload

import pandas as pd

from gaitmap.utils._types import _Hashable


class _ScoresDict(TypedDict):
    precision: float
    recall: float
    f1_score: float


@overload
def recall_score(
    matches_df: dict[_Hashable, pd.DataFrame], *, zero_division: Literal["warn", 0, 1] = "warn"
) -> dict[_Hashable, float]: ...


@overload
def recall_score(matches_df: pd.DataFrame, *, zero_division: Literal["warn", 0, 1] = "warn") -> float: ...


def recall_score(matches_df, *, zero_division: Literal["warn", 0, 1] = "warn"):
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
    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
    recall_score
        This is a float, if the input is just a single dataframe or a dictionary, if the input is a dictionary of
        dataframes.

    See Also
    --------
    gaitmap.evaluation_utils.evaluate_segmented_stride_list: Generate matched_df from stride lists


    """
    return _score_matches(matches_df, ("recall",), zero_division)


@overload
def precision_score(
    matches_df: dict[_Hashable, pd.DataFrame], *, zero_division: Literal["warn", 0, 1] = "warn"
) -> dict[_Hashable, float]: ...


@overload
def precision_score(matches_df: pd.DataFrame, *, zero_division: Literal["warn", 0, 1] = "warn") -> float: ...


def precision_score(matches_df, *, zero_division: Literal["warn", 0, 1] = "warn"):
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
    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
    precision_score
        This is a float, if the input is just a single dataframe or a dictionary, if the input is a dictionary of
        dataframes.

    See Also
    --------
    gaitmap.evaluation_utils.evaluate_segmented_stride_list: Generate matched_df from stride lists

    """
    return _score_matches(matches_df, ("precision",), zero_division)


@overload
def f1_score(
    matches_df: dict[_Hashable, pd.DataFrame], *, zero_division: Literal["warn", 0, 1] = "warn"
) -> dict[_Hashable, float]: ...


@overload
def f1_score(matches_df: pd.DataFrame, *, zero_division: Literal["warn", 0, 1] = "warn") -> float: ...


def f1_score(matches_df, *, zero_division: Literal["warn", 0, 1] = "warn"):
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
    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
    f1_score
        This is a float, if the input is just a single dataframe or a dictionary, if the input is a dictionary of
        dataframes.

    See Also
    --------
    gaitmap.evaluation_utils.evaluate_segmented_stride_list: Generate matched_df from stride lists

    """
    return _score_matches(matches_df, ("f1_score",), zero_division)


@overload
def precision_recall_f1_score(
    matches_df: dict[str, pd.DataFrame], *, zero_division: Literal["warn", 0, 1] = "warn"
) -> dict[str, _ScoresDict]: ...


@overload
def precision_recall_f1_score(
    matches_df: pd.DataFrame, *, zero_division: Literal["warn", 0, 1] = "warn"
) -> _ScoresDict: ...


def precision_recall_f1_score(matches_df, *, zero_division: Literal["warn", 0, 1] = "warn"):
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
    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.

    Returns
    -------
    score_metrics : {"precision": precision, "recall": recall, "f1_score": f1_score} or
    {"s_1": {"precision": precision, ... }, "s_2": {"recall": recall, ... }, "s_3": {"f1_score": f1_score, ... }}
        Dictionary of precision scores or a dictionary with the keys as the sensor names and the values as dictionaries
        with the scores.

    See Also
    --------
    gaitmap.evaluation_utils.evaluate_segmented_stride_list: Generate matched_df from stride lists

    """
    return _score_matches(matches_df, ("precision", "recall", "f1_score"), zero_division)


def _score_matches(matches_df, metrics, zero_division):
    if isinstance(matches_df, dict):
        return {name: _score_matches(frame, metrics, zero_division) for name, frame in matches_df.items()}

    counts = matches_df["match_type"].value_counts()
    tp, fp, fn = (int(counts.get(label, 0)) for label in ("tp", "fp", "fn"))
    ratios = {
        "precision": (tp, tp + fp),
        "recall": (tp, tp + fn),
        "f1_score": (2 * tp, 2 * tp + fp + fn),
    }
    scores = {
        metric: _calculate_score(
            *ratios[metric], zero_division=zero_division, caller_function_name=metric.removesuffix("_score")
        )
        for metric in metrics
    }
    return scores if len(metrics) > 1 else scores[metrics[0]]


def _calculate_score(a, b, *, zero_division, caller_function_name):
    try:
        return a / b
    except ZeroDivisionError as e:
        if zero_division == "warn":
            warnings.warn(
                f"Zero division happened while calculating the {caller_function_name} score. Returning 0",
                UserWarning,
            )
            return 0

        if zero_division in [0, 1]:
            return zero_division

        raise ValueError('"zero_division" must be set to "warn", 0 or 1!') from e
