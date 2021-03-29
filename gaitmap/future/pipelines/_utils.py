"""Some helper to work with the format the results of GridSearches and CVs."""
from __future__ import annotations
import numbers
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING

import joblib
import numpy as np
from sklearn.model_selection import ParameterGrid

if TYPE_CHECKING:
    from gaitmap.future.pipelines import SimplePipeline


def _aggregate_final_results(results: List) -> Dict:
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


def _normalize_score_results(scores: List, prefix="", single_score_key="score"):
    """Creates a scoring dictionary based on the type of `scores`"""
    if isinstance(scores[0], dict):
        # multimetric scoring
        return {prefix + k: v for k, v in _aggregate_final_results(scores).items()}
    # single
    return {prefix + single_score_key: scores}


def _prefix_para_dict(params_dict, prefix="pipeline__"):
    if not params_dict:
        return None
    return {prefix + k: v for k, v in params_dict.items()}


def _split_hyper_and_pure_parameters(
    parameter_grid: ParameterGrid, pure_parameters: Optional[List[str]]
) -> List[Tuple[Optional[Dict], Optional[Dict]]]:
    candidates = list(parameter_grid)
    if pure_parameters is None:
        return [(c, None) for c in candidates]
    split_candidates = []
    for c in candidates:
        tmp = {}
        for k in list(c.keys()):
            if k in pure_parameters:
                tmp[k] = c.pop(k)
        split_candidates.append((c or None, tmp or None))
    return split_candidates


def _check_safe_run(pipeline: SimplePipeline, *args, **kwargs):
    """Run the pipeline and check that run behaved as expected."""
    before_paras = pipeline.get_params()
    before_paras_hash = joblib.hash(before_paras)
    output: SimplePipeline = pipeline.run(*args, **kwargs)
    after_paras = pipeline.get_params()
    after_paras_hash = joblib.hash(after_paras)
    if not before_paras_hash == after_paras_hash:
        raise ValueError(
            "Running the pipeline did modify the parameters of the pipeline. "
            "This must not happen to make sure individual runs of the pipeline are independent.\n\n"
            "This usually happens, when you use an algorithm object as a parameter to your pipeline. "
            "In this case, make sure you call `algo_object.clone()` on the algorithm object before using "
            "it in the run method"
        )
    if not isinstance(output, type(pipeline)):
        raise ValueError(
            "The `run` method of the pipeline must return `self` or in rare cases a new instance of the "
            "pipeline itself. "
            "But the return value had the type {}".format(type(output))
        )
    if not output._action_is_applied:
        raise ValueError(
            "Running the pipeline did not set any results on the output. "
            "Make sure the `run` method sets the result values as expected as class attributes and all "
            "names of result attributes have a trailing `_` to mark them as such."
        )
    return output
