"""Some helper to work with the format the results of GridSearches and CVs."""
from __future__ import annotations
import numbers
from typing import List, TYPE_CHECKING

import joblib
import numpy as np

if TYPE_CHECKING:
    from gaitmap.future.pipelines import SimplePipeline


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


def check_safe_run(pipeline: SimplePipeline, *args, **kwargs):
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
            "This is usually happens, when you use a algorithm object as parameter to your pipeline. "
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
