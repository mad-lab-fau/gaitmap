"""Base Classes for custom pipelines."""
from typing import TypeVar, Dict, Union

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
            The structure of the data will depend on the dataset.

        Returns
        -------
        self
            The class instance with all result attributes populated

        """
        raise NotImplementedError()

    def score(self, dataset_single: Dataset) -> Union[float, Dict[str, float]]:
        """Calculate performance of the pipeline on a dataset with reference information.

        This is an optional method and does not need to be implemented in many cases.
        Usually stand-a-lone functions are better suited as scorer.

        A typical score method will call `self.run(dataset_single)` and then compare the results with reference values
        also available on the dataset.

        Parameters
        ----------
        dataset_single
            A instance of a :class:`gaitmap.future.dataset.Dataset` containing only a single datapoint.
            The structure of the data and the available reference information will depend on the dataset.

        Returns
        -------
        score
            A float or dict of float quantifying the quality of the pipeline on the provided data.
            A higher score is always better.

        """


class OptimizablePipeline(SimplePipeline):
    """Pipeline with custom ways to optimize and/or train input parameters.

    OptimizablePipelines are expected to implement a concrete way to train internal models or optimize parameters.
    This should not be reimplementation of GridSearch or similar methods.
    For this :class:`gaitmap.future.pipelines.GridSearch` should be used directly.

    It is important that `self_optimize` only modifies input parameters of the pipeline.
    This means, if a parameter is optimized, by `self_optimize` it should be named in the `__init__` and should be
    exportable when calling `pipeline.get_params`.
    It is also possible to optimize nested parameters.
    For example, if the pipeline takes a :class:`~gaitmap.stride_segmentation.DtwTemplate` as input, `self_optimize`
    can modify the template (or create a new DTW Template class).
    In any case, you should make sure that all optimized parameters are still there if you call `.clone()` on the
    optimized pipeline.
    """

    def self_optimize(self: Self, dataset: Dataset, **kwargs) -> Self:
        """Optimize the input parameter of the pipeline using any logic.

        This method can be used to adapt the input parameters (values provided in the init) based on any data driven
        heuristic.

        Note that the optimizations must only modify the input parameters (aka `self.clone` should retain the
        optimization results).

        Parameters
        ----------
        dataset
            A instance of a :class:`gaitmap.future.dataset.Dataset` containing one or multiple data points that can
            be used for training.
            The structure of the data and the available reference information will depend on the dataset.
        kwargs
            Additional parameter required for the optimization process.

        Returns
        -------
        self
            The class instance with optimized input parameters.

        """
        raise NotImplementedError()
