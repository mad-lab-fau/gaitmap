"""Helper to score pipelines."""
from __future__ import annotations

import numbers
import warnings
from traceback import format_exc
from typing import Tuple, Union, Dict, TYPE_CHECKING, List, Callable

import numpy as np
from typing_extensions import Literal

from gaitmap.future.dataset import Dataset

if TYPE_CHECKING:
    from gaitmap.future.pipelines._pipelines import SimplePipeline

_ERROR_SCORE_TYPE = Union[Literal["raise"], numbers.Number]  # noqa: invalid-name
_SCORE_TYPE = List[Union[Dict[str, numbers.Number], numbers.Number]]  # noqa: invalid-name
_AGG_SCORE_TYPE = Union[Dict[str, numbers.Number], numbers.Number]  # noqa: invalid-name
_SINGLE_SCORE_TYPE = Union[Dict[str, np.ndarray], np.ndarray]  # noqa: invalid-name


class GaitmapScorer:
    """A scorer to score multiple data points of a dataset and average the results.

    Parameters
    ----------
    score_func
        The callable that is used to score each data point
    kwargs
        Additional arguments that might be used by the scorer.
        These are ignored for the base scorer.

    """

    def __init__(self, score_func, **kwargs):
        self._kwargs = kwargs
        self._score_func = score_func

    def __call__(
        self, pipeline: SimplePipeline, data: Dataset, error_score: _ERROR_SCORE_TYPE
    ) -> Tuple[_AGG_SCORE_TYPE, _SINGLE_SCORE_TYPE]:
        """Score the pipeline with the provided data.

        Returns
        -------
        agg_scores
            The average scores over all datapoints
        single_scores
            The scores for each individual data-point

        """
        return self._score(pipeline=pipeline, data=data, error_score=error_score)

    def aggregate(self, scores: np.ndarray) -> float:  # noqa: no-self-use
        """Aggregate the scores of each data point."""
        return np.mean(scores)

    def _score(
        self, pipeline: SimplePipeline, data: Dataset, error_score: _ERROR_SCORE_TYPE
    ) -> Tuple[_AGG_SCORE_TYPE, _SINGLE_SCORE_TYPE]:
        scores = []
        for d in data:
            try:
                # We need to clone here again, to make sure that the run for each data point is truly independent.
                scores.append(self._score_func(pipeline.clone(), d))
            except Exception:  # noqa: broad-except
                if error_score == "raise":
                    raise
                scores.append(error_score)
                warnings.warn(
                    f"Scoring failed for data point: {d.groups}. "
                    f"The score of this data point will be set to {error_score}. Details: \n"
                    f"{format_exc()}",
                    UserWarning,
                )
        return _aggregate_scores(scores, self.aggregate)


def _passthrough_scoring(pipeline: SimplePipeline, dataset_single: Dataset):
    """Call the score method of the pipeline to score the input."""
    return pipeline.score(dataset_single)


def _aggregate_scores(scores: _SCORE_TYPE, agg_method: Callable) -> Tuple[_AGG_SCORE_TYPE, _SINGLE_SCORE_TYPE]:
    """Invert result dict of and apply aggregation method to each score output."""
    # We need to go through all scores and check if one is a dictionary.
    # Otherwise it might be possible that the values were caused by an error and hence did not return a dict as
    # expected.
    for s in scores:
        if isinstance(s, dict):
            score_names = s.keys()
            break
    else:
        return agg_method(scores), np.asarray(scores)
    inv_scores = {}
    agg_scores = {}
    # Invert the dict and calculate the mean per score:
    for key in score_names:
        # If the the scorer raised an error, there will only be a single value. This value will be used for all
        # scores then
        score_array = np.asarray([score[key] if isinstance(score, dict) else score for score in scores])
        inv_scores[key] = score_array
        agg_scores[key] = agg_method(score_array)
    return agg_scores, inv_scores
