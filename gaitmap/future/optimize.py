from collections import defaultdict
from functools import partial
from typing import Dict, Any, Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.ma import MaskedArray
from scipy.stats import rankdata
from sklearn.model_selection import ParameterGrid

from gaitmap.base import _BaseSerializable
from gaitmap.future.dataset import Dataset
from gaitmap.future.pipelines import SimplePipeline


def _score(pipeline: SimplePipeline, data: Dataset, parameters: Dict[str, Any]):
    pipeline = pipeline.set_params(**parameters)
    return pipeline.score(dataset=data)


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

    def score(self, datasets):
        return self.optimized_pipeline_.score(datasets)


class GridSearch(Optimize):
    parameter_grid: Optional[ParameterGrid]
    pipeline: Optional[SimplePipeline]
    n_jobs: Optional[int]
    refit: Union[bool, str]
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
        n_jobs: Optional[int] = None,
        refit: Union[bool, str] = True,
        pre_dispatch: Union[int, str] = "n_jobs",
    ):
        self.parameter_grid = parameter_grid
        self.n_jobs = n_jobs
        self.refit = refit
        self.pre_dispatch = pre_dispatch
        super().__init__(pipeline=pipeline)

    def optimize(self, dataset: Dataset, **kwargs):
        self.dataset = dataset
        candidate_params = list(self.parameter_grid)
        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)
        with parallel:
            out = parallel(
                delayed(_score)(pipeline=self.pipeline.clone(), data=dataset, parameters=paras)
                for paras in candidate_params
            )
        # We check here if all results are dicts. If yes, we have a multimetric scorer, if not, they all must be numeric
        # values and we just have a single scorer. Mixed cases will raise an error
        if all(isinstance(t, dict) for t in out):
            self.multi_metric_ = True
        elif all(isinstance(t, (int, float)) for t in out):
            self.multi_metric_ = False
        else:
            raise ValueError("The scorer must return either a dictionary of numeric values or a single numeric value.")

        results = self._format_results(candidate_params, out, multi_metric=self.multi_metric_)

        if self.refit:
            if self.multi_metric_ is True and (not isinstance(self.refit, str) or self.refit not in results):
                raise ValueError(
                    "If multi-metric scoring is used, `refit` must be a str specifying the score that should be used "
                    "to select the best result."
                )

            self.best_index_ = results["rank_{}".format(self.refit)].argmin()
            self.best_score_ = results[self.refit][self.best_index_]
            self.best_params_ = results["params"][self.best_index_]

            self.optimized_pipeline_ = self.pipeline.clone().set_params(**self.best_params_)

        self.gs_results_ = results

        return self

    def _format_results(self, candidate_params, out, multi_metric=False):
        # This function is adapted based on sklearns `BaseSearchCV`

        n_candidates = len(candidate_params)
        results = {}

        if multi_metric:
            # Invert the dict and calculate the mean per score:
            df = pd.DataFrame.from_records(out)
            for c in df.columns:
                tmp = df[c].to_numpy()
                results[c] = tmp
                results["rank_{}".format(c)] = np.asarray(rankdata(-tmp, method="min"), dtype=np.int32)
        else:
            out = np.asarray(out)
            results["score"] = out
            results["rank_score"] = np.asarray(rankdata(-out, method="min"), dtype=np.int32)

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
