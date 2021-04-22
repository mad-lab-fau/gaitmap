"""Higher level wrapper to run training and parameter optimizations."""
import time
import warnings
from collections import defaultdict
from contextlib import nullcontext
from functools import partial
from itertools import product
from tempfile import TemporaryDirectory
from typing import Dict, Any, Optional, Union, Callable, Iterator, List

import joblib
import numpy as np
from joblib import Parallel, delayed, Memory
from numpy.ma import MaskedArray
from scipy.stats import rankdata
from sklearn.model_selection import ParameterGrid, BaseCrossValidator, check_cv

from gaitmap.base import BaseAlgorithm
from gaitmap.future.dataset import Dataset
from gaitmap.future.pipelines._pipelines import SimplePipeline, OptimizablePipeline
from gaitmap.future.pipelines._score import _score, _optimize_and_score
from gaitmap.future.pipelines._scorer import GaitmapScorer, _ERROR_SCORE_TYPE, _validate_scorer
from gaitmap.future.pipelines._utils import (
    _aggregate_final_results,
    _prefix_para_dict,
    _split_hyper_and_pure_parameters,
    _normalize_score_results,
)
from gaitmap.utils.exceptions import PotentialUserErrorWarning


class BaseOptimize(BaseAlgorithm):
    """Base class for all optimizer."""

    pipeline: SimplePipeline

    dataset: Dataset

    optimized_pipeline_: SimplePipeline

    _action_method = "optimize"

    def optimize(self, dataset: Dataset, **optimize_params):
        """Apply some form of optimization on the the input parameters of the pipeline."""
        raise NotImplementedError()

    def run(self, datapoint: Dataset):
        """Run the optimized pipeline.

        This is a wrapper to contain API compatibility with `SimplePipeline`.
        """
        return self.optimized_pipeline_.run(datapoint)

    def safe_run(self, datapoint: Dataset):
        """Call the safe_run method of the optimized pipeline.

        This is a wrapper to contain API compatibility with `SimplePipeline`.
        """
        return self.optimized_pipeline_.safe_run(datapoint)

    def score(self, datapoint: Dataset):
        """Execute score on the optimized pipeline.

        This is a wrapper to contain API compatibility with `SimplePipeline`.
        """
        return self.optimized_pipeline_.score(datapoint)


class Optimize(BaseOptimize):
    """Run a generic self-optimization on the pipeline.

    This is a simple wrapper for pipelines that already implement a `self_optimize` method.
    This wrapper can be used to ensure that these algorithms can be optimized with the same interface as other
    optimization methods and can hence be used in methods like cross-validation.

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

    def optimize(self, dataset: Dataset, **optimize_params):
        """Run the self-optimization defined by the pipeline.

        The optimized version of the pipeline is stored as `self.optimized_pipeline_`.

        Parameters
        ----------
        dataset
            An instance of a :class:`~gaitmap.future.dataset.Dataset` containing one or multiple data points that can
            be used for optimization.
            The structure of the data and the available reference information will depend on the dataset.
        optimize_params
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
        # record the hash of the pipeline to make an educated guess if the optimization works
        pipeline = self.pipeline.clone()
        before_hash = joblib.hash(pipeline)
        optimized_pipeline = pipeline.self_optimize(dataset, **optimize_params)
        if not isinstance(optimized_pipeline, pipeline.__class__):
            raise ValueError(
                "Calling `self_optimize` did not return an instance of the pipeline itself! "
                "Normally this method should return `self`."
            )
        # We clone the optimized pipeline again, to make sure that only changes to the input parameters are kept.
        optimized_pipeline = optimized_pipeline.clone()
        after_hash = joblib.hash(optimized_pipeline)
        if before_hash == after_hash:
            # If the hash didn't change the object didn't change.
            # Something might have gone wrong.
            warnings.warn(
                "Optimizing the pipeline doesn't seem to have changed the parameters of the pipeline. "
                "This could indicate an implementation error of the `self_optimize` method.",
                PotentialUserErrorWarning,
            )
        self.optimized_pipeline_ = optimized_pipeline
        return self


class GridSearch(BaseOptimize):
    """Perform a GridSearch over various parameters.

    This scores the pipeline for every combination of data-points in the provided dataset and parameter combinations
    in the `parameter_grid`.
    The scores over the entire dataset are then aggregated for each para combination.
    By default this aggregation is a simple average.

    Note, that this is different to how GridSearch works in many other cases:
    Usually, the performance parameter would be calculated on all data-points at once.
    Here, each data-point represents an entire participant or gait-recording (depending on the dataset).
    Therefore, the pipeline and the scoring method are expected to provide a result/score per data-point in the dataset.
    Note, that it is still open to your interpretation, what you consider a datapoint in the context of your analysis.
    The run method of the pipeline can still process multiple e.g. gaittests in a loop and generate a single output,
    if you consider a single participant one datapoint.

    Parameters
    ----------
    pipeline
        The pipeline object to optimize
    parameter_grid
        A sklearn parameter grid to define the search space.
    scoring
        A callable that can score a single data point given a pipeline.
        This function should return either a single score or a dictionary of scores.
        If scoring is `None` the default `score` method of the pipeline is used instead.

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
        If `scoring` returns returns a dictionary of score values, this must be a str corresponding to the name of the
        score that should be used to rank the results.
        If False, the respective result attributes will not be populated.
        If multiple parameter combinations have the same score, the one tested first will be used.
        Otherwise, higher values are always considered better.
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
        The format of this dictionary is designed to be directly passed into the `pd.DataFrame` constructor.
        Each column then represents the result for one set of parameters

        The dictionary contains the following entries:

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
            The individual scores per datapoint for each parameter combination.
            This is a list of values with the `len(dataset)`.
        data_labels
            A list of data labels in the order the single score values are provided.
            These can be used to associate the `single_score` values with a certain data-point.
    optimized_pipeline_
        An instance of the input pipeline with the best parameter set.
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
    multimetric_
        If the scorer returned multiple scores

    """

    parameter_grid: ParameterGrid
    scoring: Optional[Union[Callable, GaitmapScorer]]
    n_jobs: Optional[int]
    return_optimized: Union[bool, str]
    pre_dispatch: Union[int, str]
    error_score: _ERROR_SCORE_TYPE

    gs_results_: Dict[str, Any]
    best_params_: Dict
    best_index_: int
    best_score_: float
    multimetric_: bool

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
        scoring = _validate_scorer(self.scoring, self.pipeline)

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)
        # We use a similar structure as sklearns GridSearchCV here, but instead of calling something equivalent to
        # `fit_score`, we call `score`, which just applies and scores the pipeline on the entirety of our dataset as
        # we do not need a "train" step.
        # Our main loop just loops over all parameter combis and the `_score` function then applies the para combi to
        # the pipeline and scores the resulting pipeline on the dataset, by passing the entire dataset and the
        # pipeline to the scorer.
        # Looping over the individual datapoints in the dataset and aggregating the scores is handled by the scorer
        # itself.
        # If not explicitly changed the scorer is an instance of `GaitmapScorer` that wraps the actual `scoring`
        # function provided by the user.
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
        # We check here if all results are dicts. We only check the dtype of the first value, as the scorer should
        # have handled issues with non uniform cases already.
        first_test_score = results[0]["scores"]
        self.multimetric_ = isinstance(first_test_score, dict)
        _validate_return_optimized(self.return_optimized, self.multimetric_, first_test_score)

        results = self._format_results(
            list(self.parameter_grid),
            results,
        )

        if self.return_optimized:
            return_optimized = "score"
            if self.multimetric_ and isinstance(self.return_optimized, str):
                return_optimized = self.return_optimized
            self.best_index_ = results["rank_{}".format(return_optimized)].argmin()
            self.best_score_ = results[return_optimized][self.best_index_]
            self.best_params_ = results["params"][self.best_index_]
            # We clone twice, in case one of the params was itself an algorithm.
            self.optimized_pipeline_ = self.pipeline.clone().set_params(**self.best_params_).clone()

        self.gs_results_ = results

        return self

    def _format_results(self, candidate_params, out):  # noqa: no-self-use
        """Format the final result dict.

        This function is adapted based on sklearns `BaseSearchCV`
        """
        n_candidates = len(candidate_params)
        out = _aggregate_final_results(out)

        results = {}

        scores_dict = _normalize_score_results(out["scores"])
        single_scores_dict = _normalize_score_results(out["single_scores"])
        for c, v in scores_dict.items():
            results[c] = v
            results["rank_{}".format(c)] = np.asarray(rankdata(-v, method="min"), dtype=np.int32)
            results["single_{}".format(c)] = single_scores_dict[c]

        results["data_labels"] = out["data_labels"]
        results["score_time"] = out["score_time"]

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


class GridSearchCV(BaseOptimize):
    """Exhaustive (Hyper)Parameter search using a cross validation based score to optimize pipeline parameters.

    This class follows as much as possible the interface of :func:`~sklearn.model_selection.GridSearchCV`.
    If the gaitmap documentation is missing some information, the respective documentation of sklearn might be helpful.

    Compared to the sklearn implementation this method uses a couple of gaitmap specific optimizations and
    quality-of-life improvements.

    Parameters
    ----------
    pipeline
        A gaitmap pipeline implementing `self_optimize`.
    parameter_grid
        A sklearn parameter grid to define the search space for the grid search.
    scoring
        A callable that can score a single data point given a pipeline.
        This function should return either a single score or a dictionary of scores.
        If scoring is `None` the default `score` method of the pipeline is used instead.

        .. note:: If scoring returns a dictionary, `return_optimized` must be set to the name of the score that
                  should be used for ranking.
    return_optimized
        If True, a pipeline object with the overall best params is created  and reoptimized using all provided data
        as input.
        The optimized pipeline object is stored as `optimized_pipeline_`.
        If `scoring` returns returns a dictionary of score values, this must be a str corresponding to the name of the
        score that should be used to rank the results.
        If False, the respective result attributes will not be populated.
        If multiple parameter combinations have the same mean score over all CV folds, the one tested first will be
        used.
        Otherwise, higher mean values are always considered better.
    cv
        An integer specifying the number of folds in a K-Fold cross-validation or a valid cross validation helper.
        The default (`None`) will result in a 5-fold cross validation.
        For further inputs check the sklearn documentation.
    pure_parameter_names
        .. warning: Do not use this option unless you fully understand it!

        A list of parameter names (named in the `parameter_grid`) that do not effect training aka are not
        Hyperparameters.
        This information can be used for massive performance improvements, as the training does not need to be
        repeated, if one of these parameters changes.
        However, setting it incorrectly can lead to hard to detect errors in the final results.

        For more information on this approach see the :ref:`evaluation guide <algorithm_evaluation>`.
    return_train_score
        If True the performance on the train score is returned in addition to the test score performance.
        Note, that this increases the runtime.
        If `True`, the fields `train_score`, and `train_score_single` are available in the results.
    verbose
        Control the verbosity of information printed during the optimization (larger number -> higher verbosity).
        At the moment this will only effect the caching done, when `pure_parameter_names` are provided.
    n_jobs
        The number of parallel jobs.
        The default (`None`) means 1 job at the time, hence, no parallel computing.
        -1 means as many as logical processing cores.
        One job is created per cv + para combi combintion.
    pre_dispatch
        The number of jobs that should be pre dispatched.
        For an explanation see the documentation of :class:`~sklearn.model_selection.GridSearchCV`
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
    cv_results_
        A dictionary summarizing all results of the gridsearch.
        The format of this dictionary is designed to be directly passed into the `pd.DataFrame` constructor.
        Each column then represents the result for one set of parameters.

        The dictionary contains the following entries:

        param_{parameter_name}
            The value of a respective parameter.
        params
            A dictionary representing all parameters.
        mean_test_score / mean_test_{scorer_name}
            The average test score over all folds.
            If a single score is used for scoring, then the generic name "score" is used.
            Otherwise, multiple columns with the name of the respective scorer exist.
        std_test_score / std_test_{scorer_name}
            The std of the test scores over all folds.
        rank_test_score / rank_{scorer_name}
            The rank of the mean test score assuming higher values are better.
        split{n}_test_score / split{n}_test_{scorer_name}
            The performance on the test set in fold n.
        split{n}_test_single_score / split{n}_test_single_{scorer_name}
            The performance in fold n on every single datapoint in the test set.
        split{n}_test_data_labels
            The ids of the datapoints used in the test set of fold n.
        mean_train_score / mean_train_{scorer_name}
            The average train score over all folds.
        std_train_score / std_train_{scorer_name}
            The std of the train scores over all folds.
        split{n}_train_score / split{n}_train_{scorer_name}
            The performance on the train set in fold n.
        rank_train_score / rank_{scorer_name}
            The rank of the mean train score assuming higher values are better.
        split{n}_train_single_score / split{n}_train_single_{scorer_name}
            The performance in fold n on every single datapoint in the train set.
        split{n}_train_data_labels
            The ids of the datapoints used in the train set of fold n.
        mean_{optimize/score}_time
            Average time over all folds spent for optimization and scoring, respectively.
        std_{optimize/score}_time
            Standard deviation of the optimize/score times over all folds.

    optimized_pipeline_
        An instance of the input pipeline with the best parameter set.
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
    multimetric_
        If the scorer returned multiple scores
    final_optimize_time_
        Time spent to perform the final optimization on all data.
        This is only available if `return_optimized` is not False.

    """

    pipeline: OptimizablePipeline
    parameter_grid: ParameterGrid
    scoring: Optional[Union[Callable, GaitmapScorer]]
    return_optimized: Union[bool, str]
    cv: Optional[Union[int, BaseCrossValidator, Iterator]]
    pure_parameter_names: Optional[List[str]]
    return_train_score: bool
    verbose: int
    n_jobs: Optional[int]
    pre_dispatch: Union[int, str]
    error_score: _ERROR_SCORE_TYPE

    cv_results_: Dict[str, Any]
    best_params_: Dict
    best_index_: int
    best_score_: float
    multimetric_: bool
    final_optimize_time_: float

    def __init__(
        self,
        pipeline: OptimizablePipeline,
        parameter_grid: ParameterGrid,
        *,
        scoring: Optional[Union[Callable, GaitmapScorer]] = None,
        return_optimized: Union[bool, str] = True,
        cv: Optional[Union[int, BaseCrossValidator, Iterator]] = None,
        pure_parameter_names: Optional[List[str]] = None,
        return_train_score: bool = False,
        verbose: int = 0,
        n_jobs: Optional[int] = None,
        pre_dispatch: Union[int, str] = "n_jobs",
        error_score: _ERROR_SCORE_TYPE = np.nan,
    ):
        self.pipeline = pipeline
        self.parameter_grid = parameter_grid
        self.scoring = scoring
        self.return_optimized = return_optimized
        self.cv = cv
        self.pure_parameter_names = pure_parameter_names
        self.return_train_score = return_train_score
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score

    def optimize(self, dataset: Dataset, *, groups=None, **optimize_params):  # noqa: arguments-differ
        self.dataset = dataset
        scoring = _validate_scorer(self.scoring, self.pipeline)

        cv = check_cv(self.cv, None, classifier=True)
        n_splits = cv.get_n_splits(dataset, groups=groups)

        # We need to wrap our pipeline for a consistent interface.
        # In the future we might be able to allow objects with optimizer Interface as input directly.
        optimizer = Optimize(self.pipeline)

        # For each para combi, we separate the pure parameters (parameters that do not effect the optimization) and
        # the hyperparameters.
        # This allows for massive caching optimizations in the `_optimize_and_score`.
        parameters = list(self.parameter_grid)
        split_parameters = _split_hyper_and_pure_parameters(parameters, self.pure_parameter_names)
        parameter_prefix = "pipeline__"

        # To enable the pure parameter performance improvement, we need to create a joblib cache in a temp dir that
        # is deleted after the run.
        # We only allow a temporary cache here, because the method that is cached internally is generic and the cache
        # might not be correctly invalidated, if GridSearchCv is called with a different pipeline or when the
        # pipeline itself is modified.
        tmp_dir_context = nullcontext()
        if self.pure_parameter_names:
            tmp_dir_context = TemporaryDirectory("joblib_gaitmap_cache")
        with tmp_dir_context as cachedir:
            tmp_cache = Memory(cachedir, verbose=self.verbose) if cachedir else None

            parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)
            # We use a similar structure to sklearns GridSearchCv here (see GridSearch for more info).
            with parallel:
                # Evaluate each parameter combination
                out = parallel(
                    delayed(_optimize_and_score)(
                        optimizer.clone(),
                        dataset,
                        scoring,
                        train,
                        test,
                        optimize_params=optimize_params,
                        hyperparameters=_prefix_para_dict(hyper_paras, parameter_prefix),
                        pure_parameters=_prefix_para_dict(pure_paras, parameter_prefix),
                        return_train_score=self.return_train_score,
                        return_parameters=False,
                        return_data_labels=True,
                        return_times=True,
                        error_score=self.error_score,
                        memory=tmp_cache,
                    )
                    for (cand_idx, (hyper_paras, pure_paras)), (split_idx, (train, test)) in product(
                        enumerate(split_parameters), enumerate(cv.split(dataset, groups=groups))
                    )
                )
        results = self._format_results(parameters, n_splits, out)
        self.cv_results_ = results

        first_test_score = out[0]["test_scores"]
        self.multimetric_ = isinstance(first_test_score, dict)
        _validate_return_optimized(self.return_optimized, self.multimetric_, first_test_score)
        if self.return_optimized:
            return_optimized = "score"
            if self.multimetric_ and isinstance(self.return_optimized, str):
                return_optimized = self.return_optimized
            self.best_index_ = results["rank_test_{}".format(return_optimized)].argmin()
            self.best_score_ = results["mean_test_{}".format(return_optimized)][self.best_index_]
            self.best_params_ = results["params"][self.best_index_]
            # We clone twice, in case one of the params was itself an algorithm.
            best_optimizer = Optimize(self.pipeline.clone().set_params(**self.best_params_).clone())
            final_optimize_start_time = time.time()
            optimize_params_clean = optimize_params or {}
            self.optimized_pipeline_ = best_optimizer.optimize(dataset, **optimize_params_clean).optimized_pipeline_
            self.final_optimize_time_ = final_optimize_start_time - time.time()

        return self

    def _format_results(self, candidate_params, n_splits, out, more_results=None):  # noqa: MC0001
        """Format the final result dict.

        This function is adapted based on sklearns `BaseSearchCV`.
        """
        n_candidates = len(candidate_params)
        out = _aggregate_final_results(out)

        results = dict(more_results or {})

        def _store_non_numeric(key_name: str, array):
            """Store non-numeric scores/times to the cv_results_."""
            # We avoid performing any sort of conversion into numpy arrays as this might modify the dtypes.
            # Instead we use list comprehension to do the reshaping.
            # The result is a list of lists and not a numpy array.
            #
            # Note that the results were produced by iterating first by splits and then by parameters.
            # We do the same here, but directly transpose the results, as we need to access the data per parameters
            # afterwards.
            iterable_array = iter(array)
            array = [[next(iterable_array) for _ in range(n_splits)] for _ in range(n_candidates)]
            # "Transpose" the array
            array = map(list, zip(*array))

            for split_idx, split in enumerate(array):
                # Uses closure to alter the results
                results["split{}_{}".format(split_idx, key_name)] = split

        def _store(key_name: str, array, weights=None, splits=False, rank=False):
            """Store numeric scores/times to the cv_results_."""
            # When iterated first by splits, then by parameters
            # We want `array` to have `n_candidates` rows and `n_splits` cols.
            array = np.array(array, dtype=np.float64).reshape(n_candidates, n_splits)

            if splits:
                for split_idx in range(n_splits):
                    # Uses closure to alter the results
                    results["split{}_{}".format(split_idx, key_name)] = array[:, split_idx]
            array_means = np.average(array, axis=1, weights=weights)
            results["mean_{}".format(key_name)] = array_means

            if key_name.startswith(("train_", "test_")) and np.any(~np.isfinite(array_means)):
                warnings.warn(
                    f"One or more of the {key_name.split('_')[0]} scores are non-finite: {array_means}",
                    category=UserWarning,
                )
            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(np.average((array - array_means[:, np.newaxis]) ** 2, axis=1, weights=weights))
            results["std_{}".format(key_name)] = array_stds

            if rank:
                results["rank_{}".format(key_name)] = np.asarray(rankdata(-array_means, method="min"), dtype=np.int32)

        _store("optimize_time", out["optimize_time"])
        _store("score_time", out["score_time"])
        _store_non_numeric("test_data_labels", out["test_data_labels"])
        _store_non_numeric("train_data_labels", out["train_data_labels"])
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
                param_results["param_%s" % name][cand_idx] = value

        results.update(param_results)
        # Store a list of param dicts at the key 'params'
        results["params"] = candidate_params

        test_scores_dict = _normalize_score_results(out["test_scores"])
        test_single_scores_dict = _normalize_score_results(out["test_single_scores"])
        if self.return_train_score:
            train_scores_dict = _normalize_score_results(out["train_scores"])
            train_single_scores_dict = _normalize_score_results(out["train_single_scores"])

        for scorer_name in test_scores_dict:
            # Computed the (weighted) mean and std for test scores alone
            _store("test_{}".format(scorer_name), test_scores_dict[scorer_name], splits=True, rank=True, weights=None)
            _store_non_numeric("test_single_{}".format(scorer_name), test_single_scores_dict[scorer_name])
            if self.return_train_score:
                _store("train_{}".format(scorer_name), train_scores_dict[scorer_name], splits=True)
                _store_non_numeric("train_single_{}".format(scorer_name), train_single_scores_dict[scorer_name])

        return results


def _validate_return_optimized(return_optimized, multi_metric, results):
    """Check if `return_optimize` fits to the multimetric output of the scorer."""
    if multi_metric is True:
        # In a multimetric case, return_optimized must either be False or a string
        if return_optimized and (not isinstance(return_optimized, str) or return_optimized not in results):
            raise ValueError(
                "If multi-metric scoring is used, `return_optimized` must be a str specifying the score that "
                "should be used to select the best result."
            )
    else:
        if isinstance(return_optimized, str):
            warnings.warn(
                "You set `return_optimized` to the name of a scorer, but the provided scorer only produces a "
                "single score. `return_optimized` is set to True."
            )
