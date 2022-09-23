"""Classes representing data transformations as preprocessing for different algorithms."""
from gaitmap.data_transform._base import (
    BaseTransformer,
    GroupedTransformer,
    IdentityTransformer,
    TrainableTransformerMixin,
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
    "TrainableMinMaxScaler",
    "MinMaxScaler",
    "FixedScaler",
    "AbsMaxScaler",
    "TrainableAbsMaxScaler",
    "StandardScaler",
    "TrainableStandardScaler",
    "TrainableTransformerMixin",
    "BaseTransformer",
    "BaseFilter",
    "ButterworthFilter",
    "GroupedTransformer",
]
