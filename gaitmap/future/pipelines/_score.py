"""This is a modified version of sklearn fit and score functionality.

The original code is licenced under BSD-3: https://github.com/scikit-learn/scikit-learn
"""

from __future__ import annotations

import numbers
import time
import warnings
from traceback import format_exc
from typing import Tuple, Union, Dict, Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from sklearn import clone
from sklearn.exceptions import FitFailedWarning
from sklearn.pipeline import Pipeline
from typing_extensions import Literal

from gaitmap.future.dataset import Dataset
from gaitmap.future.pipelines._scorer import GaitmapScorer

if TYPE_CHECKING:
    from gaitmap.future.pipelines._optimize import Optimize


# TODO: Split in multiple optimize_and_score into a score func -> Score can be used for Gridseearch


def _score(
    pipeline: Pipeline,
    dataset: Dataset,
    scorer: GaitmapScorer,
    parameters: Dict,
    return_parameters=False,
    return_data_labels=False,
    return_times=False,
    candidate_progress=None,
    error_score=np.nan,
):
    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        pipeline = pipeline.set_params(**cloned_parameters)

    start_time = time.time()
    agg_scores, single_scores = scorer(pipeline, dataset, error_score)
    score_time = time.time() - start_time

    result = {}
    result["scores"] = agg_scores
    result["single_scores"] = single_scores
    if return_times:
        result["score_time"] = score_time
    if return_data_labels:
        result["data"] = dataset.groups
    if return_parameters:
        result["parameters"] = parameters
    return result


def _optimize_and_score(
    optimizer: Optimize,
    dataset: Dataset,
    scorer: GaitmapScorer,
    train: Optional[ArrayLike],
    test: Optional[ArrayLike],
    verbose,
    parameters,
    optimize_params: Optional[Dict],
    return_train_score=False,
    return_parameters=False,
    return_data_labels=False,
    return_times=False,
    return_estimator=False,
    split_progress=None,
    candidate_progress=None,
    error_score=np.nan,
):

    """Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    optimizer : estimator object implementing 'fit'
        The object to use to fit the data.

    dataset : array-like of shape (n_samples, n_features)
        The data to fit.

    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.

        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.

        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.

    train : array-like of shape (n_train_samples,)
        Indices of training samples.

    test : array-like of shape (n_test_samples,)
        Indices of test samples.

    verbose : int
        The verbosity level.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

    parameters : dict or None
        Parameters to be set on the estimator.

    optimize_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

    return_train_score : bool, default=False
        Compute and return score on training set.

    return_parameters : bool, default=False
        Return parameters that has been used for the estimator.

    split_progress : {list, tuple} of int, default=None
        A list or tuple of format (<current_split_id>, <total_num_of_splits>).

    candidate_progress : {list, tuple} of int, default=None
        A list or tuple of format
        (<current_candidate_id>, <total_number_of_candidates>).

    return_n_test_samples : bool, default=False
        Whether to return the ``n_test_samples``.

    return_times : bool, default=False
        Whether to return the fit/score times.

    return_estimator : bool, default=False
        Whether to return the fitted estimator.

    Returns
    -------
    result : dict with the following attributes
        train_scores : dict of scorer name -> float
            Score on training set (for all the scorers),
            returned only if `return_train_score` is `True`.
        test_scores : dict of scorer name -> float
            Score on testing set (for all the scorers).
        n_test_samples : int
            Number of test samples.
        fit_time : float
            Time spent for fitting in seconds.
        score_time : float
            Time spent for scoring in seconds.
        parameters : dict or None
            The parameters that have been evaluated.
        estimator : estimator object
            The fitted estimator.
        fit_failed : bool
            The estimator failed to fit.
    """
    if not isinstance(error_score, numbers.Number) and error_score != "raise":
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    progress_msg = ""
    if verbose > 2:
        if split_progress is not None:
            progress_msg = f" {split_progress[0]+1}/{split_progress[1]}"
        if candidate_progress and verbose > 9:
            progress_msg += f"; {candidate_progress[0]+1}/" f"{candidate_progress[1]}"

    if verbose > 1:
        if parameters is None:
            params_msg = ""
        else:
            sorted_keys = sorted(parameters)  # Ensure deterministic o/p
            params_msg = ", ".join(f"{k}={parameters[k]}" for k in sorted_keys)
    if verbose > 9:
        start_msg = f"[CV{progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    optimize_params = optimize_params if optimize_params is not None else {}

    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        optimizer = optimizer.set_params(**cloned_parameters)

    start_time = time.time()

    train_set = None
    if train:
        train_set = dataset[train]
    test_set = dataset
    if test:
        test_set = dataset[test]

    result = {}
    try:
        optimizer.optimize(train_set, **optimize_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == "raise":
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(scorer, dict):
                test_scores = {name: error_score for name in scorer}
                if return_train_score:
                    train_scores = test_scores.copy()
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
            warnings.warn(
                "Estimator fit failed. The score on this train-test"
                " partition for these parameters will be set to %f. "
                "Details: \n%s" % (error_score, format_exc()),
                FitFailedWarning,
            )
        result["fit_failed"] = True
    else:
        result["fit_failed"] = False

        fit_time = time.time() - start_time
        test_scores = scorer(optimizer, test_set, error_score)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores = scorer(optimizer, train_set, scorer, error_score)

    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = f"[CV{progress_msg}] END "
        result_msg = params_msg + (";" if params_msg else "")
        if verbose > 2 and isinstance(test_scores, dict):
            for scorer_name in sorted(test_scores):
                result_msg += f" {scorer_name}: ("
                if return_train_score:
                    scorer_scores = train_scores[scorer_name]
                    result_msg += f"train={scorer_scores:.3f}, "
                result_msg += f"test={test_scores[scorer_name]:.3f})"
        result_msg += f" total time={logger.short_format_time(total_time)}"

        # Right align the result_msg
        end_msg += "." * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        print(end_msg)

    result["test_scores"] = test_scores
    if return_train_score:
        result["train_scores"] = train_scores
    if return_times:
        result["fit_time"] = fit_time
        result["score_time"] = score_time
    if return_data_labels:
        result["train_data"] = train_set.groups
        result["test_data"] = test_set.groups
    if return_parameters:
        result["parameters"] = parameters
    if return_estimator:
        result["estimator"] = optimizer
    return result
