import numbers
from typing import Dict, List, Union, Tuple

import numpy as np
import pandas as pd


def _aggregate_scores(
    scores: List[Union[Dict[str, numbers.Number], numbers.Number]]
) -> Tuple[Union[Dict[str, numbers.Number], numbers.Number], Union[str, Dict[str, np.ndarray], np.ndarray]]:
    # TODO: This doesn't work if scoring fails.
    if isinstance(scores[0], dict):
        # Invert the dict and calculate the mean per score:
        df = pd.DataFrame.from_records(scores)
        means = df.mean()
        return means.to_dict(), {k: v.to_numpy() for k, v in df.iteritems()}
    return np.mean(scores), np.asarray(scores)


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
