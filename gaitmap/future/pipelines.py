"""Base Classes for custom pipelines."""
from typing import TypeVar, Dict, Union

import numpy as np
import pandas as pd

from gaitmap.base import _BaseSerializable
from gaitmap.future.dataset import Dataset

Self = TypeVar("Self", bound="SimplePipeline")


class SimplePipeline(_BaseSerializable):
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

    def score_single(self, dataset_single: Dataset) -> Union[float, Dict[str, float]]:
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

    def score(self, dataset: Dataset) -> Union[float, Dict[str, float]]:
        """Return the performance metric as average over multiple datapoints in the dataset.

        Parameters
        ----------
        dataset
            A instance of a :class:`gaitmap.future.dataset.Dataset` containing multiple datapoint.

        Returns
        -------
        score
            The performance value(s) as average over the scores of each datapoint.
        """
        scores = []
        for d in dataset:
            scores.append(self.score_single(d))
        if isinstance(scores[0], dict):
            # Invert the dict and calculate the mean per score:
            df = pd.DataFrame.from_records(scores)
            means = df.mean()
            return means.to_dict()
        return np.mean(scores)
