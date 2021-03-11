import numbers
from typing import Dict, List, Union, Tuple

import numpy as np


def _aggregate_scores_mean(
    scores: List[Union[Dict[str, numbers.Number], numbers.Number]]
) -> Tuple[Union[Dict[str, numbers.Number], numbers.Number], Union[str, Dict[str, np.ndarray], np.ndarray]]:
    # We need to go through all scores and check if one is a dictionary.
    # Otherwise it might be possible that the values were caused by an error and hence did not return a dict as
    # expected.
    for s in scores:
        if isinstance(s, dict):
            break
    else:
        return np.mean(scores), np.asarray(scores)
    inv_scores = {}
    agg_scores = {}
    # Invert the dict and calculate the mean per score:
    for key in s:
        # If the the scorer raised an error, there will only be a single value. This value will be used for all
        # scores then
        score_array = np.asarray([score[key] if isinstance(score, dict) else score for score in scores])
        inv_scores[key] = score_array
        agg_scores[key] = np.mean(score_array)
    return agg_scores, inv_scores


def _aggregate_final_results(results: List):
    """Aggregate the list of dict to dict of np ndarray/list.

    Modified based on sklearn.model_selection._validation._aggregate_score_dicts


    Parameters
    ----------

    results : list of dict
        List of dicts of the results for all scorers. This is a flat list,
        assumed originally to be of row major order.

    Example
    -------

    >>> results = [{'a': 1, 'b':10}, {'a': 2, 'b':2}, {'a': 3, 'b':3}, {'a': 10, 'b': 10}]
    >>> _aggregate_final_results(results)                        # doctest: +SKIP
    {'a': array([1, 2, 3, 10]),
     'b': array([10, 2, 3, 10])}
    """
    return {
        key: np.asarray([score[key] for score in results])
        if isinstance(results[0][key], numbers.Number)
        else [score[key] for score in results]
        for key in results[0]
    }
