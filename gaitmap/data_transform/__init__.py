"""Classes representing data transformations as preprocessing for different algorithms."""

from gaitmap.data_transform._base import (
    BaseTransformer,
    ChainedTransformer,
    GroupedTransformer,
    IdentityTransformer,
    ParallelTransformer,
    TrainableTransformerMixin,
)
from gaitmap.data_transform._feature_transform import (
    BaseSlidingWindowFeatureTransform,
    Resample,
    SlidingWindowGradient,
    SlidingWindowMean,
    SlidingWindowStd,
    SlidingWindowVar,
)
from gaitmap.data_transform._filter import BaseFilter, ButterworthFilter
from gaitmap.data_transform._scaler import (
    AbsMaxScaler,
    FixedScaler,
    MinMaxScaler,
    StandardScaler,
    TrainableAbsMaxScaler,
    TrainableMinMaxScaler,
    TrainableStandardScaler,
)

__all__ = [
    "AbsMaxScaler",
    "BaseFilter",
    "BaseSlidingWindowFeatureTransform",
    "BaseTransformer",
    "ButterworthFilter",
    "ChainedTransformer",
    "FixedScaler",
    "GroupedTransformer",
    "IdentityTransformer",
    "MinMaxScaler",
    "ParallelTransformer",
    "Resample",
    "SlidingWindowGradient",
    "SlidingWindowMean",
    "SlidingWindowStd",
    "SlidingWindowVar",
    "StandardScaler",
    "TrainableAbsMaxScaler",
    "TrainableMinMaxScaler",
    "TrainableStandardScaler",
    "TrainableTransformerMixin",
]
