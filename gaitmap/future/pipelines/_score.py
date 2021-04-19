"""This is a modified version of sklearn fit and score functionality.

The original code is licenced under BSD-3: https://github.com/scikit-learn/scikit-learn
"""

from __future__ import annotations

import numbers
import time
from typing import Dict, Optional, TYPE_CHECKING, Union, Tuple, List, Any, Type

import joblib
import numpy as np
from joblib import Memory
from typing_extensions import TypedDict

from gaitmap.future.dataset import Dataset
from gaitmap.future.pipelines._pipelines import SimplePipeline
from gaitmap.future.pipelines._scorer import GaitmapScorer, _ERROR_SCORE_TYPE, _SINGLE_SCORE_TYPE, _AGG_SCORE_TYPE
from gaitmap.future.pipelines._utils import _get_nested_paras, _clone_parameter_dict

if TYPE_CHECKING:
    from gaitmap.future.pipelines._optimize import BaseOptimize  # noqa: cyclic-import


class ScoreResults(TypedDict, total=False):
    """Type representing results of _score."""

    scores: _AGG_SCORE_TYPE
    single_scores: _SINGLE_SCORE_TYPE
    score_time: float
    data_labels: List[Union[str, Tuple[str, ...]]]
    parameters: Optional[Dict[str, Any]]


class OptimizeScoreResults(TypedDict, total=False):
    """Type representing results of _score_and_optimize."""

    test_scores: _AGG_SCORE_TYPE
    test_single_scores: _SINGLE_SCORE_TYPE
    train_scores: _AGG_SCORE_TYPE
    train_single_scores: _SINGLE_SCORE_TYPE
    score_time: float
    optimize_time: float
    train_data_labels: List[Union[str, Tuple[str, ...]]]
    test_data_labels: List[Union[str, Tuple[str, ...]]]
    parameters: Optional[Dict[str, Any]]
    optimizer: BaseOptimize


def _score(
    pipeline: SimplePipeline,
    dataset: Dataset,
    scorer: GaitmapScorer,
    parameters: Optional[Dict[str, Any]],
    return_parameters=False,
    return_data_labels=False,
    return_times=False,
    error_score: _ERROR_SCORE_TYPE = np.nan,
) -> ScoreResults:
    """Set parameters and return score.

    Parameters
    ----------
    pipeline
        An instance of a gaitmap pipeline
    dataset
        An instance of a gaitmap dataset with multiple data points.
    scorer
        A scorer that calculates a score by running the pipeline on each data point and then aggregates the results.
    parameters : dict of valid parameters for the pipeline
        The parameters that should be set for the pipeline before scoring
    return_parameters
        If the parameter value that was inputted should be added to the result dict
    return_data_labels
        If the names of the data points should be added to the result dict
    return_times
        If the time required to score the dataset should be added to the result dict
    error_score
        The value that should be used if scoring fails for a specific data point.
        This can be any numeric value (including nan and inf) or "raises".
        If it is "raises", the scoring error is raised instead of ignored.
        In all other cases a warning is displayed.
        Note, that if the value is set to np.nan, the aggregated value over multiple data points will also be nan,
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
            "(Hint: if using 'raise', please make sure that it has been spelled correctly.)"
        )

    if parameters is not None:
        # clone after setting parameters in case any parameters are estimators (like pipeline steps).
        parameters = _clone_parameter_dict(parameters)

        pipeline = pipeline.set_params(**parameters)

    start_time = time.time()
    agg_scores, single_scores = scorer(pipeline, dataset, error_score)
    score_time = time.time() - start_time

    result: ScoreResults = {"scores": agg_scores, "single_scores": single_scores}
    if return_times:
        result["score_time"] = score_time
    if return_data_labels:
        result["data_labels"] = dataset.groups
    if return_parameters:
        result["parameters"] = parameters
    return result


def _optimize_and_score(  # noqa: too-many-branches
    optimizer: BaseOptimize,
    dataset: Dataset,
    scorer: GaitmapScorer,
    train: np.ndarray,
    test: np.ndarray,
    *,
    optimize_params: Optional[Dict] = None,
    hyperparameters: Optional[Dict] = None,
    pure_parameters: Optional[Dict] = None,
    return_train_score=False,
    return_optimizer=False,
    return_parameters=False,
    return_data_labels=False,
    return_times=False,
    error_score: _ERROR_SCORE_TYPE = np.nan,
    memory: Optional[Memory] = None,
) -> OptimizeScoreResults:
    """Optimize and score the optimized pipeline on the train and test data, respectively.

    This method is aware of the differences between hyperparameters and normal (pure) parameters.
    This can be used to cache the results of training, as the training results should only depend on hyperparameters,
    but not on normal parameters.
    The provided `memory` instance is used to perform this caching step.

    Note, that caching in this context should only be performed in the context of a single call to e.g. GridSearchCV
    and the cache should be deleted afterwards to avoid cache leak.
    Therefore, the cachedir should ideally be set to a random tmp dir by the caller.
    """
    if memory is None:
        memory = Memory(None)
    # clone after setting parameters in case any parameters are estimators (like pipeline steps).
    hyperparameters = _clone_parameter_dict(hyperparameters)
    pure_parameters = _clone_parameter_dict(pure_parameters)

    optimize_params_clean: Dict = optimize_params or {}

    train_set = dataset[train]
    test_set = dataset[test]

    start_time = time.time()
    optimizer = _cached_optimize(optimizer, train_set, hyperparameters, pure_parameters, memory, optimize_params_clean)
    optimize_time = time.time() - start_time

    # Now we set the remaining paras.
    # Because, we need to set the parameters on the optimized pipeline and not the input pipeline we strip the
    # naming prefix.
    striped_paras = _get_nested_paras(pure_parameters, "pipeline")
    optimizer.optimized_pipeline_.set_params(**striped_paras)
    # We also set the parameters of the input pipeline to make it seem that all parameters were set from the
    # beginning.
    optimizer = optimizer.set_params(**pure_parameters)

    agg_scores, single_scores = scorer(optimizer.optimized_pipeline_, test_set, error_score)
    score_time = time.time() - optimize_time - start_time

    if return_train_score:
        train_agg_scores, train_single_scores = scorer(optimizer.optimized_pipeline_, train_set, error_score)

    result: OptimizeScoreResults = {"test_scores": agg_scores, "test_single_scores": single_scores}
    if return_train_score:
        result["train_scores"] = train_agg_scores
        result["train_single_scores"] = train_single_scores
    if return_times:
        result["score_time"] = score_time
        result["optimize_time"] = optimize_time
    if return_data_labels:
        # Note we always return the train data attribute as it is interesting information independent of the train
        # score and has 0 runtime impact.
        result["train_data_labels"] = train_set.groups
        result["test_data_labels"] = test_set.groups
    if return_optimizer:
        # This is the actual trained optimizer. This means that `optimizer.optimized_pipeline_` contains the actual
        # instance of the trained pipeline.
        result["optimizer"] = optimizer
    if return_parameters:
        result["parameters"] = {**hyperparameters, **pure_parameters} or None
    return result


def _cached_optimize(
    optimizer: BaseOptimize, data: Dataset, hyperparameters: Dict, pure_parameters: Dict, memory: Memory, optimize_paras
):
    """Set parameters and optimize a pipeline and cache the optimization result.

    This method will cache the training as long as the hyperparameters stay the same.
    Changing the pure parameters will not invalidate the cache.

    """
    # We do not set all paras right away, we first create a cached optimize function that only has the hyper
    # parameters as input.
    # This allows to cache the train results, if the _optimize_and_score is called multiple times with the same hyper
    # parameters.
    # To be sure that nothing "bad" happens here, we also pass in the pipeline class itself to invalidate the cache,
    # in case a completely different pipeline/algorithm is optimized.
    # Ideally the `memory` object used here should only be used once.
    # E.g. for a single a GridSearchCV.
    def cachable_optimize(
        opti: Type[BaseOptimize], hyperparas: Dict[str, Any], data: Dataset, optimize_params: Dict
    ) -> BaseOptimize:
        _ = opti
        return optimizer.set_params(**hyperparas).optimize(data, **optimize_params)

    optimize_func = memory.cache(cachable_optimize)
    # Optimization must never modify pure parameters, or we have a problem.
    # We check that by calculating the hash of all pure parameters before and after the optimization.
    opti_paras = optimizer.get_params()
    pure_para_subset = {k: opti_paras[k] for k in pure_parameters}
    pure_para_hash = joblib.hash(pure_para_subset)
    pipeline_pure_para_hash = joblib.hash(_get_nested_paras(pure_para_subset, "pipeline"))

    # This is the actual call to train the optimizer:
    optimizer = optimize_func(type(optimizer), hyperparameters, data, optimize_paras)

    opti_paras = optimizer.get_params()
    optimized_pipeline_paras = optimizer.optimized_pipeline_.get_params()
    # We check that the pure parameters on the optimize object haven't changed and that the pure parameters belonging
    # to the pipeline have not changed in the `optimized_pipeline`.
    # Note, that the first case will never happen with gaitmap native Optimizers, but could happen for custom
    # optimizers.
    if pipeline_pure_para_hash != joblib.hash(
        {k: optimized_pipeline_paras[k] for k in _get_nested_paras(pure_parameters, "pipeline")}
    ) or pure_para_hash != joblib.hash({k: opti_paras[k] for k in pure_parameters}):
        raise ValueError(
            "Optimizing the pipeline modified a parameter marked as `pure`. "
            "This must not happen. "
            "Double check your optimize implementation and the list of pure parameters."
        )

    return optimizer
