from collections import defaultdict
from functools import partial
from typing import Dict, Any, Optional, Union, Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.ma import MaskedArray
from scipy.stats import rankdata
from sklearn.model_selection import ParameterGrid

from gaitmap.base import _BaseSerializable
from gaitmap.future.dataset import Dataset
from gaitmap.future.pipelines import SimplePipeline


def _aggregate_scores(scores):
    if isinstance(scores[0], dict):
        # Invert the dict and calculate the mean per score:
        df = pd.DataFrame.from_records(scores)
        means = df.mean()
        return means.to_dict(), {k: v.to_numpy() for k, v in df.iteritems()}
    return np.mean(scores), scores

# TODO: If we have a score single method, we could cache it in the context of GridSearch nested in a cv
def _score(pipeline: SimplePipeline, scoring: Callable, data: Dataset, parameters: Dict[str, Any]):
    pipeline = pipeline.set_params(**parameters)
    pipeline = pipeline.clone()
    # TODO: Perform aggregation of performance here: Aka mean and maybe weighting?
    #       This would allow to return all individual scores for each "dataset_single" object or even all fitted result
    #       objects for each score.
    scores = []
    for d in data:
        # We clone again here, in case any of the parameters were algorithms themself or the score method of the
        # pipeline does strange things.
        scores.append(scoring(pipeline, dataset_single=d))
    return _aggregate_scores(scores)

def _passthrough_scoring(pipeline, dataset_single):
    return pipeline.score(dataset_single)


class Optimize(_BaseSerializable):
    pipeline: Optional[SimplePipeline]

    dataset: Dataset

    optimized_pipeline_: SimplePipeline

    def __init__(
        self,
        pipeline: Optional[SimplePipeline] = None,
    ):
        self.pipeline = pipeline

    def optimize(self, dataset: Dataset, **kwargs):
        self.dataset = dataset
        if not hasattr(self.pipeline, "self_optimize"):
            raise ValueError()
        self.optimized_pipeline_ = self.pipeline.clone().self_optimize(dataset, **kwargs)
        return self

    def run_optimized(self, dataset_single):
        return self.optimized_pipeline_.run(dataset_single)

    def score(self, dataset_single):
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
        with parallel:
            out = parallel(
                delayed(_score)(pipeline=self.pipeline.clone(), scoring=scoring, data=dataset, parameters=paras)
                for paras in candidate_params
            )
        mean_scores, data_point_scores = zip(*out)
        # We check here if all results are dicts. If yes, we have a multimetric scorer, if not, they all must be numeric
        # values and we just have a single scorer. Mixed cases will raise an error
        if all(isinstance(t, dict) for t in mean_scores):
            self.multi_metric_ = True
        elif all(isinstance(t, (int, float)) for t in mean_scores):
            self.multi_metric_ = False
        else:
            raise ValueError("The scorer must return either a dictionary of numeric values or a single numeric value.")

        results = self._format_results(
            candidate_params, mean_scores, data_point_scores, multi_metric=self.multi_metric_
        )

        if self.multi_metric_ is True and (not isinstance(self.rank_scorer, str) or self.rank_scorer not in results):
            raise ValueError(
                "If multi-metric scoring is used, `rank_scorer` must be a str specifying the score that should be used "
                "to select the best result."
            )

        rank_score = self.rank_scorer or "score"
        self.best_index_ = results["rank_{}".format(rank_score)].argmin()
        self.best_score_ = results[rank_score][self.best_index_]
        self.best_params_ = results["params"][self.best_index_]
        # We clone twice, in case one of the params was itself a algorithm.
        self.optimized_pipeline_ = self.pipeline.clone().set_params(**self.best_params_).clone()

        self.gs_results_ = results

        return self

    def _format_results(self, candidate_params, mean_scores, data_point_scores=None, multi_metric=False):
        # This function is adapted based on sklearns `BaseSearchCV`

        n_candidates = len(candidate_params)
        results = {}

        if multi_metric:
            # Invert the dict and calculate the mean per score:
            df = pd.DataFrame.from_records(mean_scores)
            for c, v in df.iteritems():
                v = v.to_numpy()
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


__all__ = ["GridSearch", "Optimize"]
