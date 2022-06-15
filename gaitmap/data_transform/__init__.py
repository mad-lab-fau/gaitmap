"""Classes representing data transformations as preprocessing for different algorithms."""
from gaitmap.data_transform._scaler import (
    AbsMaxScaler,
    BaseTransformer,
    FixedScaler,
    GroupedTransformer,
    IdentityTransformer,
    MinMaxScaler,
    StandardScaler,
    TrainableAbsMaxScaler,
    TrainableMinMaxScaler,
    TrainableStandardScaler,
    TrainableTransformerMixin,
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
    "IdentityTransformer",
    "StandardScaler",
    "TrainableStandardScaler",
]
