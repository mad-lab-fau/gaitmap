"""Base Classes for custom pipelines."""
from typing import TypeVar, Dict, Union

import numpy as np
import pandas as pd

from gaitmap.base import BaseAlgorithm
from gaitmap.future.dataset import Dataset

Self = TypeVar("Self", bound="SimplePipeline")


class SimplePipeline(BaseAlgorithm):
    """Baseclass for all custom pipelines.

    To create your own custom pipeline, subclass this class and implement `run`.
    """

    dataset_single: Dataset

    _action_method = "run"

    def run(self: Self, dataset_single: Dataset) -> Self:
        """Run the pipeline.

        Parameters
        ----------
        dataset_single
            A instance of a :class:`gaitmap.future.dataset.Dataset` containing only a single datapoint.

        Returns
        -------
        self
            The class instance with all result attributes populated

        """
        raise NotImplementedError()


class ScoreableMixin:
    """A pipeline that can calculate a performance score for a given input.

    This should be used as Mixin if you plan to use to use `GridSearch` (or similar methods) to improve potential
    parameter of your `SimplePipeline` sub-class.
    `GridSearch` will then use the `score_single` (or rather the `score`) method to rank the results (by default a
    higher score is better).

    To use it, subclass the pipeline and implement `run` and `score_single`.
    If you need more control on how performance values are averaged over multiple data-points (participants,
    gait-tests, ...), you can also overwrite `score`.

    Note, `score_single` and `score` must return either a numeric value or a dictionary of numeric values.
    """

    def score(self, dataset_single: Dataset) -> Union[float, Dict[str, float]]:
        """Return a performance metric for the result of a single datapoint.

        Parameters
        ----------
        dataset_single
            A instance of a :class:`gaitmap.future.dataset.Dataset` containing only a single datapoint.

        Returns
        -------
        score
            The performance value(s)

        """
        raise NotImplementedError()
