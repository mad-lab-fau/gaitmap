"""Base Classes for custom pipelines."""
from typing import TypeVar

import numpy as np

from gaitmap.base import _BaseSerializable
from gaitmap.future import Dataset

Self = TypeVar("Self", bound="SimplePipeline")


class SimplePipeline(_BaseSerializable):
<<<<<<< HEAD
    def run(self, datasets_single):
        raise NotImplementedError()

    def score_single(self, datasets_single):
        raise NotImplementedError()

    def score(self, datasets):
=======
    """Baseclass for all custom pipelines.

    To create your own custom pipeline, subclass this class and implement `run` and optionally `score_single`.
    You can also overwrite `score`, in case you need to customize how score values are averaged over mutliple
    data points.
    """

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

    def score_single(self, dataset_single: Dataset) -> float:
        """Return a performance metric for the result of a single datapoint.

        Parameters
        ----------
        dataset_single
            A instance of a :class:`gaitmap.future.dataset.Dataset` containing only a single datapoint.

        Returns
        -------
        score
            The performance value

        """
        raise NotImplementedError()

    def score(self, dataset: Dataset) -> float:
        """Return the performance metric as average over multiple datapoints in the dataset.

        Parameters
        ----------
        dataset
            A instance of a :class:`gaitmap.future.dataset.Dataset` containing multiple datapoint.

        Returns
        -------
        score
            The performance value as average over the scores of each datapoint.
        """
>>>>>>> c4a6d0f (Added docstrings and correct typing)
        scores = []
        for d in datasets:
            scores.append(self.score_single(d))
        return np.mean(scores)
