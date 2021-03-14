"""Higher level wrapper to run training and parameter optimizations."""
import numbers
import warnings
from collections import defaultdict
from functools import partial
from typing import Dict, Any, Optional, Union, Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.ma import MaskedArray
from scipy.stats import rankdata
from sklearn.model_selection import ParameterGrid

from gaitmap.base import BaseAlgorithm
from gaitmap.future.dataset import Dataset
from gaitmap.future.pipelines._pipelines import SimplePipeline, OptimizablePipeline
from gaitmap.future.pipelines._score import _score
from gaitmap.future.pipelines._scorer import GaitmapScorer, _ERROR_SCORE_TYPE, _validate_scorer
from gaitmap.future.pipelines._utils import _aggregate_final_results


class _BaseOptimize(BaseAlgorithm):
    pipeline: SimplePipeline

    dataset: Dataset

    optimized_pipeline_: SimplePipeline

    _action_method = "optimize"

    def optimize(self, dataset: Dataset, **kwargs):
        """Apply some form of optimization on the the input parameters of the pipeline."""
        raise NotImplementedError()

    def run(self, dataset_single):
        """Run the optimized pipeline.

        This is a wrapper to contain API compatibility with `SimplePipeline`.
        """
        return self.optimized_pipeline_.run(dataset_single)

    def score(self, dataset_single):
        """Execute score on the optimized pipeline.

        This is a wrapper to contain API compatibility with `SimplePipeline`.
        """
        return self.optimized_pipeline_.score(dataset_single)


class Optimize(_BaseOptimize):
    """Run a generic self-optimization on the pipeline.

    This is a simple wrapper for pipelines that already implement a `self_optimize` method.
    This wrapper can be used to ensure that these algorithms can be optimized with the same interface as other
    optimization methods and can hence, be used in methods like cross-validation.

    Optimize will never modify the original pipeline, but will store a copy of the optimized pipeline as
    `optimized_pipeline_`.

    Parameters
    ----------
    pipeline
        The pipeline to optimize.
        The pipeline must implement `self_optimize` to optimize its own input parameters.

    Other Parameters
    ----------------
    dataset
        The dataset used for optimization.

    Attributes
    ----------
    optimized_pipeline_
        The optimized version of the pipeline.
        That is a copy of the input pipeline with modified params.

    """

    pipeline: OptimizablePipeline

    optimized_pipeline_: OptimizablePipeline

    def __init__(self, pipeline: OptimizablePipeline):
        self.pipeline = pipeline

    def optimize(self, dataset: Dataset, **kwargs):
        """Run the self-optimization defined by the pipeline.

        The optimized version of the pipeline is stored as `self.optimized_pipeline_`.

        Parameters
        ----------
        dataset
            A instance of a :class:`~gaitmap.future.dataset.Dataset` containing one or multiple data points that can
            be used for optimization.
            The structure of the data and the available reference information will depend on the dataset.
        kwargs
            Additional parameter for the optimization process.
            They are forwarded to `pipeline.self_optimize`.

        Returns
        -------
        self
            The class instance with all result attributes populated

        """
        self.dataset = dataset
        if not hasattr(self.pipeline, "self_optimize"):
            raise ValueError(
                "To use `Optimize` with a pipeline, the pipeline needs to implement a `self_optimize` method."
            )
        self.optimized_pipeline_ = self.pipeline.clone().self_optimize(dataset, **kwargs)
        return self


class GridSearch(_BaseOptimize):
    """Perform a GridSearch over various parameters.

    This scores the pipeline for every combination of data-points in the provided dataset and parameter combinations
    in the `parameter_grid`.
    The scores over the entire dataset are then aggregated for each para combination.
    By default this aggregation is a simple average.

    Note, that this is different to how GridSearch works in many other cases:
    Usually, the performance parameter would be calculated on all data-points at once.
    Here, each data-point represents a entire participant or gait-recording (depending on the dataset).
    Therefore, the pipeline and the scoring method are expected to provide a result/score per data-point in the dataset.

    Parameters
    ----------
    pipeline
        The pipeline object to optimize
    parameter_grid
        A sklearn parameter grid to define the search space.
    scoring
        A callable that can score a single data point given a pipeline.
        This function should return either a single score or a dictionary of scores.

        .. note:: If scoring returns a dictionary, `return_optimized` must be set to the name of the score that
                  should be used for ranking.
    n_jobs
        The number of processes that should be used to parallelize the search.
        None means 1, -1 means as many as logical processing cores.
    pre_dispatch
        The number of jobs that should be pre dispatched.
        For an explanation see the documentation of :class:`~sklearn.model_selection.GridSearchCV`
    return_optimized
        If True, a pipeline object with the overall best params is created and stored as `optimized_pipeline_`.
        If `scoring` returns multiple score values, this must be a str corresponding to the name of the score that
        should be used to rank the results.
        If False, the respective result attributes will not be populated.
    error_score
        Value to assign to the score if an error occurs during scoring.
        If set to ‘raise’, the error is raised.
        If a numeric value is given, a Warning is raised.

    Other Parameters
    ----------------
    dataset
        The dataset instance passed to the optimize method

    Attributes
    ----------
    gs_results_
        A dictionary summarizing all results of the gridsearch.
        The format of this dictionary is design to be directly passed into the `pd.DataFrame` constructor.
        Each column then represents the result for one set of parameters

        The dictionary contains the following columns:

        param_*
            The value of a respective parameter
        params
            A dictionary representing all parameters
        score / {scorer-name}
            The aggregated value of a score over all data-points.
            If a single score is used for scoring, than the generic name "score" is used.
            Otherwise multiple columns with the name of the respective scorer exist
        rank_score / rank_{scorer-name}
            A sorting for each score from the highest to the lowest value
        single_score / single_{scorer-name}
            The individual scores per datapoint for each score.
            This is a list of values with the `len(dataset)`.
        data_labels
            A list of data labels in the order the single score values are provided.
            These can be used to associate the `single_score` values with a certain data-point.
    optimized_pipeline_
        A instance of the input pipeline with the best parameter set.
        This is only available if `return_optimized` is not False.
    best_params_
        The parameter dict that resulted in the best result.
        This is only available if `return_optimized` is not False.
    best_index_
        The index of the result row in the output.
        This is only available if `return_optimized` is not False.
    best_score_
        The score of the best result.
        In a multimetric case, only the value of the scorer specified by `return_optimized` is provided.
        This is only available if `return_optimized` is not False.
    multi_metric_
        Rather the scorer returned multiple scores

    """

    parameter_grid: ParameterGrid
    scoring: Optional[Union[Callable, GaitmapScorer]]
    n_jobs: Optional[int]
    return_optimized: Optional[str]
    pre_dispatch: Union[int, str]
    error_score: _ERROR_SCORE_TYPE

    gs_results_: Dict[str, Any]
    best_params_: Dict
    best_index_: int
    best_score_: float
    multi_metric_: bool

    def __init__(
        self,
        pipeline: SimplePipeline,
        parameter_grid: ParameterGrid,
        *,
        scoring: Optional[Union[Callable, GaitmapScorer]] = None,
        n_jobs: Optional[int] = None,
        pre_dispatch: Union[int, str] = "n_jobs",
        return_optimized: Union[bool, str] = True,
        error_score: _ERROR_SCORE_TYPE = np.nan,
    ):
        self.pipeline = pipeline
        self.parameter_grid = parameter_grid
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.return_optimized = return_optimized
        self.error_score = error_score

    def optimize(self, dataset: Dataset, **_):
        """Run the GridSearch over the dataset and find the best parameter combination.

        Parameters
        ----------
        dataset
            The dataset used for optimization.

        """
        self.dataset = dataset
        scoring = _validate_scorer(self.scoring)

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)
        with parallel:
            # Evaluate each parameter combination
            results = parallel(
                delayed(_score)(
                    self.pipeline.clone(),
                    dataset,
                    scoring,
                    paras,
                    return_parameters=True,
                    return_data_labels=True,
                    return_times=True,
                    error_score=self.error_score,
                )
                for paras in self.parameter_grid
            )
        results = _aggregate_final_results(results)
        mean_scores = results["scores"]
        data_point_scores = results["single_scores"]
        # We check here if all results are dicts. If yes, we have a multimetric scorer, if not, they all must be numeric
        # values and we just have a single scorer. Mixed cases will raise an error
        if all(isinstance(v, dict) for v in mean_scores):
            self.multi_metric_ = True
            # In a multimetric case, we need to flatten the individual score dicts.
            mean_scores = _aggregate_final_results(mean_scores)
            data_point_scores = _aggregate_final_results(data_point_scores)
        elif all(isinstance(t, numbers.Number) for t in mean_scores):
            self.multi_metric_ = False
        else:
            raise ValueError("The scorer must return either a dictionary of numeric values or a single numeric value.")

        results = self._format_results(
            results["parameters"],
            mean_scores,
            data_point_scores=data_point_scores,
            data_point_names=results["data_labels"],
            multi_metric=self.multi_metric_,
        )

        self._validate_return_optimized(results)
        if self.return_optimized:
            return_optimized = "score"
            if self.multi_metric_ and self.return_optimized:
                return_optimized = self.return_optimized
            self.best_index_ = results["rank_{}".format(return_optimized)].argmin()
            self.best_score_ = results[return_optimized][self.best_index_]
            self.best_params_ = results["params"][self.best_index_]
            # We clone twice, in case one of the params was itself a algorithm.
            self.optimized_pipeline_ = self.pipeline.clone().set_params(**self.best_params_).clone()

        self.gs_results_ = results

        return self

    def _validate_return_optimized(self, results):
        """Check if `return_optimize` fits to the multimetric output of the scorer."""
        if self.multi_metric_ is True:
            # In a multimetric case, return_optimized must either be False of a string
            if self.return_optimized is True or self.return_optimized not in results:
                raise ValueError(
                    "If multi-metric scoring is used, `rank_scorer` must be a str specifying the score that should be "
                    "used to select the best result."
                )
        else:
            if isinstance(self.return_optimized, str):
                warnings.warn(
                    "You set `return_optimized` to the name of a scorer, but the provided scorer only produces a "
                    "single score. `return_optimized` is set to True."
                )

    def _format_results(  # noqa: no-self-use
        self, candidate_params, mean_scores, *, data_point_scores=None, data_point_names=None, multi_metric=False
    ):
        """Format the final result dict.

        This function is adapted based on sklearns `BaseSearchCV`
        """
        # TODO: Add time

        n_candidates = len(candidate_params)
        results = {}

        if multi_metric:
            for c, v in mean_scores.items():
                results[c] = v
                results["rank_{}".format(c)] = np.asarray(rankdata(-v, method="min"), dtype=np.int32)
            if data_point_scores:
                df_single = pd.DataFrame.from_records(data_point_scores)
                for c, v in df_single.iteritems():
                    results["single_{}".format(c)] = v.to_numpy()
        else:
            results["score"] = mean_scores
            results["rank_score"] = np.asarray(rankdata(-mean_scores, method="min"), dtype=np.int32)
            if data_point_scores:
                results["single_score"] = data_point_scores

        if data_point_names is not None:
            results["data_labels"] = data_point_names

        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(
            partial(
                MaskedArray,
                np.empty(
                    n_candidates,
                ),
                mask=True,
                dtype=object,
            )
        )
        for cand_idx, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurrence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_{}".format(name)][cand_idx] = value

        results.update(param_results)
        # Store a list of param dicts at the key 'params'
        results["params"] = candidate_params

        return results
