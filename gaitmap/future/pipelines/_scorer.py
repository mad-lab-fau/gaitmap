from __future__ import annotations

import numbers
import warnings
from traceback import format_exc
from typing import Tuple, Union, Dict, TYPE_CHECKING

import numpy as np
from typing_extensions import Literal

from gaitmap.future.dataset import Dataset
from gaitmap.future.pipelines._utils import _aggregate_scores_mean

if TYPE_CHECKING:
    from gaitmap.future.pipelines._pipelines import SimplePipeline

_ERROR_SCORE_TYPE = Union[Literal["raise"], numbers.Number]


class GaitmapScorer:
    def __init__(self, score_func, **kwargs):
        self._kwargs = kwargs
        self._score_func = score_func

    def __call__(
        self, pipeline: SimplePipeline, data: Dataset, error_score: _ERROR_SCORE_TYPE
    ) -> Tuple[Union[Dict[str, numbers.Number], numbers.Number], Union[Dict[str, np.ndarray], np.ndarray]]:
        return self._score(pipeline=pipeline, data=data, error_score=error_score)

    def aggregate(self, scores):
        return _aggregate_scores_mean(scores)

    def _score(self, pipeline: SimplePipeline, data: Dataset, error_score: _ERROR_SCORE_TYPE):
        scores = []
        for d in data:
            try:
                # We need to clone here again, to make sure that the run for each data point is truly independent.
                scores.append(self._score_func(pipeline.clone(), d))
            except Exception:
                if error_score == "raise":
                    raise
                else:
                    scores.append(error_score)
                    warnings.warn(
                        f"Scoring failed for data point: {d.groups}. "
                        f"The score of this data point will be set to {error_score}. Details: \n"
                        f"{format_exc()}",
                        UserWarning,
                    )
        return self.aggregate(scores)


def _passthrough_scoring(pipeline: SimplePipeline, dataset_single: Dataset):
    return pipeline.score(dataset_single)
