"""Classes representing data transformations as preprocessing for different algorithms."""
from gaitmap.data_transform._scaler import (
    BaseTransformer,
    MinMaxScaler,
    GroupedTransformer,
    TrainableTransformerMixin,
    FixedScaler,
    AbsMaxScaler,
    TrainableAbsMaxScaler,
    TrainableMinMaxScaler,
)

__all__ = [
    "BaseTransformer",
    "TrainableMinMaxScaler",
    "MinMaxScaler",
    "GroupedTransformer",
    "TrainableTransformerMixin",
    "FixedScaler",
    "AbsMaxScaler",
    "TrainableAbsMaxScaler",
]
