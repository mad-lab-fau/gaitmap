"""This is a modified version of sklearn fit and score functionality.

The original code is licenced under BSD-3: https://github.com/scikit-learn/scikit-learn
"""

from __future__ import annotations

import numbers
import time
import warnings
from traceback import format_exc
from typing import Dict, Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from sklearn import clone
from sklearn.exceptions import FitFailedWarning

from gaitmap.future.dataset import Dataset
from gaitmap.future.pipelines._pipelines import SimplePipeline
from gaitmap.future.pipelines._scorer import GaitmapScorer, _ERROR_SCORE_TYPE

if TYPE_CHECKING:
    from gaitmap.future.pipelines._optimize import Optimize


def _score(
    pipeline: SimplePipeline,
    dataset: Dataset,
    scorer: GaitmapScorer,
    parameters: Optional[Dict],
    return_parameters=False,
    return_data_labels=False,
    return_times=False,
    error_score: _ERROR_SCORE_TYPE = np.nan,
):
    """Set parameters and return score.

    Parameters
    ----------
    pipeline
        A instance of a gaitmap pipeline
    dataset
        A instance of a gaitmap dataset with multiple data points.
    scorer
        A scorer that calculates a score by running the pipeline on each data point and then aggregates the results.
    parameters
        The parameters that should be set for the pipeline before scoring
    return_parameters
        If the parameter value that was input should be added to the result dict
    return_data_labels
        If the names of the data points should be added to the result dict
    return_times
        If the time required to score the dataset added to the result dict
    error_score
        The value that should be used if scoring fails for a specific data point.
        This can be any numeric value (including nan and inf) or "raises".
        If it is "raises", the scoring error is raised instead of ignored.
        In all other cases a warning is displayed.
        Note, that if the value is set to np.nan, the aggreagated value over multiple data points will also be nan,
        if scoring fails for a single data point.

    Returns
    -------
    result : dict with the following attributes
        scores : dict of scorer name -> float
            Calculated scores
        scores_single : dict of scorer name -> np.ndarray
            Calculated scores for each individual data point
        score_time : float
            Time required to score the dataset
        data : List
            List of data point labels used
        parameters : dict or None
            The parameters that have been evaluated.

    """
    if not isinstance(error_score, numbers.Number) and error_score != "raise":
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    if parameters is not None:
        # clone after setting parameters in case any parameters are estimators (like pipeline steps).
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        pipeline = pipeline.set_params(**cloned_parameters)

    start_time = time.time()
    agg_scores, single_scores = scorer(pipeline, dataset, error_score)
    score_time = time.time() - start_time

    result = dict()
    result["scores"] = agg_scores
    result["single_scores"] = single_scores
    if return_times:
        result["score_time"] = score_time
    if return_data_labels:
        result["data_labels"] = dataset.groups
    if return_parameters:
        result["parameters"] = parameters
    return result


def _optimize_and_score(
    optimizer: Optimize,
    dataset: Dataset,
    scorer: GaitmapScorer,
    train: Optional[ArrayLike],
    test: Optional[ArrayLike],
    parameters,
    optimize_params: Optional[Dict],
    return_train_score=False,
    return_parameters=False,
    return_data_labels=False,
    return_times=False,
    return_estimator=False,
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
        Parameters that will be passed to ``optimizer.optimize``.

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
