"""Helper to validate/evaluate pipelines and Optimize.

TODO: We might move this to evaluation utils
"""

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import BaseCrossValidator, check_cv

from gaitmap.future.dataset import Dataset
from gaitmap.future.pipelines._optimize import BaseOptimize
from gaitmap.future.pipelines._score import _optimize_and_score
from gaitmap.future.pipelines._scorer import _ERROR_SCORE_TYPE, _validate_scorer
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
    """Evaluate a pipeline on a dataset using cross validation.

    This function follows as much as possible the interface of :func:`~sklearn.model_selection.cross_validate`.
    If the gaitmap documentation is missing some information, the respective documentation of sklearn might be helpful.

    Parameters
    ----------
    optimizable
        A optimizable class instance like `GridSearch`/`GridSearchCV` or a pipeline wrapped in an `Optimize` object.
    dataset
        A gaitmap dataset containing all information.
    groups
        Group labels for samples used by the cross validation helper, in case a grouped CV is used (e.g.
        :class:`~sklearn.model_selection.GroupKFold`).
        Check the documentation of the :class:`~gaitmap.future.dataset.Dataset` class and the respective example for
        information on how to generate group labels for gaitmap datasets.
    scoring
        A callable that can score a single data point given a pipeline.
        This function should return either a single score or a dictionary of scores.
        If scoring is `None` the default `score` method of the optimizable is used instead.
    cv
        An integer specifying the number of folds in a K-Fold cross-validation or a valid cross validation helper.
        The default (`None`) will result in a 5-fold cross validation.
        For further inputs check the sklearn documentation.
    n_jobs
        Number of jobs to run in parallel.
        One job is created per CV fold.
        The default (`None`) means 1 job at the time, hence, no parallel computing.
    verbose
        The verbosity level (larger number -> higher verbosity).
        At the moment this only effects `Parallel`.
    optimize_params
        Additional parameter that are forwarded to the `optimize` method.
    pre_dispatch
        The number of jobs that should be pre dispatched.
        For an explanation see the documentation of :class:`~joblib.Parallel`.
    return_train_score
        If True the performance on the train score is returned in addition to the test score performance.
        Note, that this increases the runtime.
        If `True`, the fields `train_data_labels`, `train_score`, and `train_score_single` are available in the results.
    return_optimizer
        If the optimized instance of the input optimizable should be returned.
        If `True`, the field `optimizer` is available in the results.
    error_score
        Value to assign to the score if an error occurs during scoring.
        If set to ‘raise’, the error is raised.
        If a numeric value is given, a Warning is raised.

    Returns
    -------
    result_dict
        Dictionary with results.
        Each element is either a list or array of length `n_folds`.
        The dictionary can be directly passed into the pandas DataFrame constructor for a better representation.

        The following fields are in the results:

        test_score / test_{scorer-name}
            The aggregated value of a score over all data-points.
            If a single score is used for scoring, than the generic name "score" is used.
            Otherwise multiple columns with the name of the respective scorer exist.
        test_single_score / test_single_{scorer-name}
            The individual scores per datapoint per fold.
            This is a list of values with the `len(train_set)`.
        test_data_labels
            A list of data labels of the train set in the order the single score values are provided.
            These can be used to associate the `single_score` values with a certain data-point.
        train_score / train_{scorer-name}
            Results for train set of each fold.
        train_single_score / train_single_{scorer-name}
            Results for individual datapoints in the train set of each fold
        train_data_labels
           The data labels for the train set.
        optimize_time
            Time required to optimize the pipeline in each fold.
        score_time
            Cumulative score time to score all data points in the test set.
        optimizer
            The optimized instances per fold.
            One instance per fold is returned.
            The optimized version of the pipeline can be obtained via the `optimized_pipeline_` attribute on the
            instance.

    """
    cv = check_cv(cv, None, classifier=True)

    scoring = _validate_scorer(scoring, optimizable.pipeline)

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
    for name in ["test_scores", "test_single_scores", "train_scores", "train_single_scores"]:
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
