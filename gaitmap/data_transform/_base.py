from copy import copy
from functools import reduce
from typing import Sequence, Optional, List, Tuple, Union, Set

import pandas as pd
from tpcp import OptimizableParameter, PureParameter, make_action_safe

from typing_extensions import Self

from base import BaseAlgorithm
from utils._types import _Hashable
from utils.datatype_helper import SingleSensorData


class BaseTransformer(BaseAlgorithm):
    """Base class for all data transformers."""

    _action_methods = ("transform",)

    transformed_data_: SingleSensorData

    data: SingleSensorData

    def transform(self, data: SingleSensorData, **kwargs) -> Self:
        """Transform the data using the transformer.

        Parameters
        ----------
        data
            A dataframe representing single sensor data.

        Returns
        -------
        self
            The instance of the transformer with the results attached

        """
        raise NotImplementedError()


class TrainableTransformerMixin:
    """Mixin for transformers with adaptable parameters."""

    def self_optimize(self, data: Sequence[SingleSensorData], **kwargs) -> Self:
        """Learn the parameters of the transformer based on provided data.

        Parameters
        ----------
        data
           A sequence of dataframes, each representing single-sensor data.
        kwargs
            Optional keword arguments

        Returns
        -------
        self
            The trained instance of the transformer

        """
        raise NotImplementedError()


class GroupedTransformer(BaseTransformer, TrainableTransformerMixin):
    """Apply specific transformations to specific groups of columns.

    Parameters
    ----------
    transformer_mapping
        List of tuples to define which transformers should be applied to which columns.
        The list should have the shape [(key, transformer), ...] where key is either the name of the column or a
        tuple of column names.
        If the transformer is trainable, its `self_optimize` method will be called, when `self_optimize` of the
        Grouped Transformer is called.
    keep_all_cols
        If `True`, columns that are not mentioned as keys in the `transformer_mapping`, will be added to the output
        unchanged.
        Otherwise, only columns that are actually transformed remain in the output.

    Attributes
    ----------
    transformed_data_
        The transformed data.

    Other Parameters
    ----------------
    data
        The data passed to the transform method.

    """

    transformer_mapping: OptimizableParameter[
        Optional[List[Tuple[Union[_Hashable, Tuple[_Hashable, ...]], BaseTransformer]]]
    ]
    keep_all_cols: PureParameter[bool]

    data: SingleSensorData

    def __init__(
        self,
        transformer_mapping: Optional[List[Tuple[Union[_Hashable, Tuple[_Hashable, ...]], BaseTransformer]]] = None,
        keep_all_cols: bool = True,
    ):
        self.transformer_mapping = transformer_mapping
        self.keep_all_cols = keep_all_cols

    def self_optimize(self, data: Sequence[SingleSensorData], **kwargs) -> Self:
        """Train all trainable transformers based on the provided data.

        ... note :: All transformers will be trained on all columns they are applied to as group.
                    This means you will get different results when using `(("col_a", "col_b"),
                    transformer())` compared to `("col_a", transformer()), ("col_b", transformer())`.
                    In the first case the transformer will be trained over both columns as one.

        Parameters
        ----------
        data
           A sequence of dataframes, each representing single-sensor data.

        Returns
        -------
        self
            The trained instance of the transformer

        """
        if self.transformer_mapping is None:
            return self

        # Check that all transformers are individual objects and not the same object multiple times
        transformer_ids = [id(k[1]) for k in self.transformer_mapping]

        if len(set(transformer_ids)) != len(transformer_ids):
            raise ValueError(
                "All transformers must be different objects when trying to optimize them. "
                "At least two transformer instances point to the same object. "
                "This can cause unexpected results. "
                "Make sure each transformer is a separate instance of a transformer class."
            )

        mapped_cols = self._validate_mapping()
        for d in data:
            self._validate(d, mapped_cols)
        trained_transformers = []
        for k, v in self.transformer_mapping:
            if isinstance(v, TrainableTransformerMixin):
                col_select = k
                if not isinstance(col_select, tuple):
                    col_select = (col_select,)
                trained_transformers.append((k, v.self_optimize([d[list(col_select)] for d in data], **kwargs)))
            else:
                trained_transformers.append((k, v))
        self.transformer_mapping = trained_transformers
        return self

    @make_action_safe
    def transform(self, data: SingleSensorData, **kwargs) -> Self:
        """Transform all data columns based on the selected transformers.

        Parameters
        ----------
        data
            A dataframe representing single sensor data.

        Returns
        -------
        self
            The instance of the transformer with the results attached

        """
        self.data = data
        if self.transformer_mapping is None:
            self.transformed_data_ = copy(data)
            return self
        mapped_cols = self._validate_mapping()
        self._validate(data, mapped_cols)
        results = []
        mapping = self.transformer_mapping
        if self.keep_all_cols:
            mapping = [*mapping, *((k, IdentityTransformer()) for k in set(data.columns) - mapped_cols)]
            mapped_cols = set(data.columns)
        for k, v in mapping:
            if not isinstance(k, tuple):
                k = (k,)
            # We clone here to make sure that we do not modify the "parameters" within the action method
            tmp = v.clone().transform(data[list(k)], **kwargs).transformed_data_
            for col in k:
                results.append(tmp[[col]])
        self.transformed_data_ = pd.concat(results, axis=1)[sorted(mapped_cols, key=list(data.columns).index)]
        return self

    def _validate_mapping(self) -> Set[_Hashable]:
        # Check that each column is only mentioned once:
        unique_k = []
        for k, _ in self.transformer_mapping:
            if not isinstance(k, tuple):
                k = (k,)
            for i in k:
                if i in unique_k:
                    raise ValueError(
                        "Each column name must only be mentioned once in the keys of `scaler_mapping`."
                        "Applying multiple transformations to the same column is not supported."
                    )
                unique_k.append(i)
        return set(unique_k)

    def _validate(self, data: SingleSensorData, selected_cols: Set[_Hashable]):  # noqa: no-self-use
        if not set(data.columns).issuperset(selected_cols):
            raise ValueError("You specified transformations for columns that do not exist. This is not supported!")


class ChainedTransformer(BaseTransformer, TrainableTransformerMixin):
    """Apply a series of transformations to the input."""

    transformer_chain: OptimizableParameter[Sequence[Union[BaseTransformer, TrainableTransformerMixin]]]

    def __init__(self, transformer_chain: Sequence[Union[BaseTransformer, TrainableTransformerMixin]]):
        self.transformer_chain = transformer_chain

    def self_optimize(self, data: Sequence[SingleSensorData], **kwargs) -> Self:
        """Optimize the chained transformers.

        Note, that each transformer only gets the transformed output of the previous trained transformer in the chain.

        Basically, training works as follows:

        1. Optimize transformer 1
        2. Transform data using optimized transformer 1
        3. Optimize transformer 2 with transformed data after transformer 1
        4. ...

        Parameters
        ----------
        kwargs
            All kwargs will be passed to **all** transformers `self_optimize` and `transform` methods called in the
            process.

        """
        data = self.data
        optimized_transformers = []

        for transformer in self.transformer_chain:
            if isinstance(transformer, TrainableTransformerMixin):
                transformer = transformer.self_optimize(data, **kwargs)
            optimized_transformers.append(transformer.clone())
            data = [transformer.transform(d, **kwargs).transformed_data_ for d in data]

        self.transformer_chain = optimized_transformers
        return self

    def transform(self, data: SingleSensorData, **kwargs) -> Self:
        self.transformed_data_ = reduce(lambda t, d: t.clone().transform(d, **kwargs), self.transformer_chain, data)
        return self


class ParallelTransformer(BaseTransformer, TrainableTransformerMixin):
    """Apply multiple different transformation to the input, resulting in multiple outputs.

    Note, all outputs of all transformers must have the same second dimension, so that they can all be combined into
    a single dataframe.
    """

    transformers: OptimizableParameter[List[Tuple[_Hashable, BaseTransformer]]]

    def __init__(self, transformers: List[Tuple[_Hashable, BaseTransformer]]):
        self.transformers = transformers

    def self_optimize(self, data: Sequence[SingleSensorData], **kwargs) -> Self:
        optimized_transformers = []
        for transformer in self.transformers:
            if isinstance(transformer, TrainableTransformerMixin):
                transformer = transformer.self_optimize(data, **kwargs)
            optimized_transformers.append(transformer.clone())

        self.transformers = optimized_transformers
        return self

    def transform(self, data: SingleSensorData, **kwargs) -> Self:
        transformed_data = {k: t.clone().transform(data, **kwargs).transformed_data_ for k, t in self.transformers}
        combined_results = pd.concat(transformed_data, axis=1)
        combined_results.columns = [f"{a}__{b}" for a, b in combined_results.columns]
        self.transformed_data_ = combined_results
        return self


class IdentityTransformer(BaseTransformer):
    """Dummy Transformer that does not modify the data and simply returns a copy of the input.

    Attributes
    ----------
    transformed_data_
        The transformed data.

    Other Parameters
    ----------------
    data
        The data passed to the transform method.

    """

    data: SingleSensorData

    def transform(self, data: SingleSensorData, **_) -> Self:
        """Transform the data (aka do nothing for this transformer).

        Parameters
        ----------
        data
            A dataframe representing single sensor data.

        Returns
        -------
        self
            The instance of the transformer with the results attached

        """
        self.data = data
        self.transformed_data_ = copy(data)
        return self
