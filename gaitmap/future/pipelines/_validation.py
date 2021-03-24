import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import check_cv

from gaitmap.future.pipelines._score import _optimize_and_score
from gaitmap.future.pipelines._scorer import _validate_scorer


def cross_validate(
    optimizable,
    dataset,
    *,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    optimize_params=None,
    pre_dispatch="2*n_jobs",
    return_train_score=False,
    return_optimizer=False,
    error_score=np.nan,
):
    """Run cross validation.

    Note: Verbose will only passed to parallel
    """

    cv = check_cv(cv, dataset, classifier=True)

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
            return_optimizer=return_optimizer,
            error_score=error_score,
        )
        for train, test in cv.split(dataset, groups=groups)
    )

    return results
