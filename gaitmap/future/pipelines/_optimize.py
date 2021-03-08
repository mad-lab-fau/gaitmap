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
from sklearn.model_selection._validation import _aggregate_score_dicts

from gaitmap.base import _BaseSerializable
from gaitmap.future.dataset import Dataset
from gaitmap.future.pipelines._pipelines import SimplePipeline, OptimizablePipeline
from gaitmap.future.pipelines._score import _score
from gaitmap.future.pipelines._scorer import GaitmapScorer, _passthrough_scoring


class Optimize(_BaseSerializable):
    pipeline: Optional[OptimizablePipeline]

    dataset: Dataset

    optimized_pipeline_: OptimizablePipeline

    def __init__(
        self,
        pipeline: Optional[OptimizablePipeline] = None,
    ):
        self.pipeline = pipeline

    def optimize(self, dataset: Dataset, **kwargs):
        """Run the self-optimization defined by the pipeline.

        The optimized version of the pipeline is stored as `self.optimized_pipeline_`

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


class GridSearch(Optimize):
    parameter_grid: Optional[ParameterGrid]
    scoring: Optional[Callable]
    pipeline: Optional[SimplePipeline]
    n_jobs: Optional[int]
    rank_scorer: Optional[str]
    pre_dispatch: Union[int, str]

    dataset: Dataset

    best_params_: Dict
    best_index_: int
    best_score_: float
    gs_results_: Dict[str, Any]
    multi_metric_: bool

    def __init__(
        self,
        pipeline: Optional[SimplePipeline] = None,
        parameter_grid: Optional[ParameterGrid] = None,
        *,
        scoring: Optional[Callable] = None,
        n_jobs: Optional[int] = None,
        rank_scorer: Optional[str] = None,
        pre_dispatch: Union[int, str] = "n_jobs",
    ):
        self.parameter_grid = parameter_grid
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.rank_scorer = rank_scorer
        self.pre_dispatch = pre_dispatch
        super().__init__(pipeline=pipeline)

    def optimize(self, dataset: Dataset, **kwargs):
        self.dataset = dataset
        candidate_params = list(self.parameter_grid)
        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)
        scoring = self.scoring
        if scoring is None:
            scoring = _passthrough_scoring
        if callable(scoring):
            scoring = GaitmapScorer(scoring)
        with parallel:
            results = parallel(
                delayed(_score)(self.pipeline.clone(), dataset, scoring, paras) for paras in candidate_params
            )
        # TODO: Create own version of aggregate_score_dicts and fix versions, where scoring failed
        results = _aggregate_score_dicts(results)
        mean_scores = results["scores"]
        data_point_scores = results["single_scores"]
        # We check here if all results are dicts. If yes, we have a multimetric scorer, if not, they all must be numeric
        # values and we just have a single scorer. Mixed cases will raise an error
        if all(isinstance(v, dict) for v in mean_scores):
            self.multi_metric_ = True
            # In a multimetric case, we need to flatten the individual score dicts.
            mean_scores = _aggregate_score_dicts(mean_scores)
            data_point_scores = _aggregate_score_dicts(data_point_scores)
        elif all(isinstance(t, numbers.Number) for t in mean_scores):
            self.multi_metric_ = False
        else:
            raise ValueError("The scorer must return either a dictionary of numeric values or a single numeric value.")

        results = self._format_results(
            candidate_params,
            mean_scores,
            data_point_scores=data_point_scores,
            data_point_names=None,
            multi_metric=self.multi_metric_,
        )

        if self.multi_metric_ is True and (not isinstance(self.rank_scorer, str) or self.rank_scorer not in results):
            raise ValueError(
                "If multi-metric scoring is used, `rank_scorer` must be a str specifying the score that should be used "
                "to select the best result."
            )
        if not self.multi_metric_ and isinstance(self.rank_scorer, str):
            warnings.warn(
                "You specified `rank_scorer`, but the provided scorer only produces a single score. "
                "`rank_scorer` is ignored."
            )

        rank_score = "score"
        if self.multi_metric_ and self.rank_scorer:
            rank_score = self.rank_scorer
        self.best_index_ = results["rank_{}".format(rank_score)].argmin()
        self.best_score_ = results[rank_score][self.best_index_]
        self.best_params_ = results["params"][self.best_index_]
        # We clone twice, in case one of the params was itself a algorithm.
        self.optimized_pipeline_ = self.pipeline.clone().set_params(**self.best_params_).clone()

        self.gs_results_ = results

        return self

    def _format_results(
        self, candidate_params, mean_scores, *, data_point_scores=None, data_point_names=None, multi_metric=False
    ):
        # This function is adapted based on sklearns `BaseSearchCV`

        n_candidates = len(candidate_params)
        results = {}

        if multi_metric:
            # Invert the dict and calculate the mean per score:
            for c, v in mean_scores.items():
                results[c] = v
                results["rank_{}".format(c)] = np.asarray(rankdata(-v, method="min"), dtype=np.int32)
            if data_point_scores:
                df_single = pd.DataFrame.from_records(data_point_scores)
                for c, v in df_single.iteritems():
                    results["single_{}".format(c)] = v.to_numpy()
        else:
            mean_scores = np.asarray(mean_scores)
            results["score"] = mean_scores
            results["rank_score"] = np.asarray(rankdata(-mean_scores, method="min"), dtype=np.int32)
            if data_point_scores:
                results["single_score"] = data_point_scores

        if data_point_names is not None:
            results["data"] = data_point_names

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
