"""Helper to validate/evaluate pipelines and Optimize.

TODO: We might move this to evaluation utils
"""

from typing import Union, Any, Dict, Optional, Iterator, Callable, Tuple, List

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import check_cv, BaseCrossValidator

from gaitmap.future.dataset import Dataset
from gaitmap.future.pipelines._optimize import BaseOptimize
from gaitmap.future.pipelines._score import _optimize_and_score
from gaitmap.future.pipelines._scorer import _validate_scorer, _ERROR_SCORE_TYPE
from gaitmap.future.pipelines._utils import _aggregate_final_results, _normalize_score_results


def cross_validate(
    optimizable: BaseOptimize,
    dataset: Dataset,
    *,
    groups: Optional[List[Union[str, Tuple[str, ...]]]] = None,
    scoring: Optional[Callable] = None,
    cv: Optional[Union[int, BaseCrossValidator, Iterator]] = None,
    n_jobs: Optional[int] = None,
    verbose: int = 0,
    optimize_params: Optional[Dict[str, Any]] = None,
    pre_dispatch: Union[str, int] = "2*n_jobs",
    return_train_score: bool = False,
    return_optimizer: bool = False,
    error_score: _ERROR_SCORE_TYPE = np.nan,
):
    """Run cross validation.

    Note: Verbose will only passed to parallel
    """
    cv = check_cv(cv, None, classifier=True)

    scoring = _validate_scorer(scoring)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    results = parallel(
        delayed(_optimize_and_score)(
            optimizable.clone(),
            dataset,
            scoring,
            train,
            test,
            optimize_params=optimize_params,
            hyperparameters=None,
            pure_parameters=None,
            return_train_score=return_train_score,
            return_times=True,
            return_data_labels=True,
            return_optimizer=return_optimizer,
            error_score=error_score,
        )
        for train, test in cv.split(dataset, groups=groups)
    )

    results = _aggregate_final_results(results)
    score_results = {}
    # Fix the formatting of all the score results
    for name in ["scores", "single_scores", "train_scores", "train_single_scores"]:
        if name in results:
            score = results.pop(name)
            prefix = ""
            if "_" in name:
                prefix = name.rsplit("_", 1)[0] + "_"
            score = _normalize_score_results(score, prefix)
            # We use a new dict here, as it is unsafe to append a dict you are iterating over
            score_results.update(score)

    results.update(score_results)
    return results
