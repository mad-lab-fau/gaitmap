from typing import Dict, Union, Tuple, List

import numpy as np
import pandas as pd

from gaitmap.base import _BaseSerializable
from gaitmap.utils._types import _Hashable
from gaitmap.utils.datatype_helper import SensorData, SingleSensorData, is_sensor_data, get_multi_sensor_names


class BaseTransformer(_BaseSerializable):
    def transform(self, data: SingleSensorData, sampling_rate_hz: float) -> SingleSensorData:
        raise NotImplementedError()


class TrainableTransformer(BaseTransformer):
    def train(self, data: SensorData, sampling_rate_hz: float):
        raise NotImplementedError()


class GroupedTransformer(TrainableTransformer):
    def __init__(self, scaler_mapping: Dict[Union[_Hashable, Tuple[_Hashable, ...]], BaseTransformer]):
        self.scaler_mapping = scaler_mapping

    def train(self, data: SensorData, sampling_rate_hz: float):
        self._validate(data)
        for k, v in self.scaler_mapping.items():
            if isinstance(v, TrainableTransformer):
                self.scaler_mapping[k] = v.train(data, sampling_rate_hz=sampling_rate_hz)
        return self

    def transform(self, data: SingleSensorData, sampling_rate_hz: float) -> SingleSensorData:
        self._validate(data)
        results = []
        for k, v in self.scaler_mapping.items():
            results.append(v.transform(data[list(k)], sampling_rate_hz=sampling_rate_hz))
        return pd.concat(results)[data.columns]

    def _validate(self, data):
        # Check that each column is only mentioned once:
        unique_k = []
        for k in self.scaler_mapping.keys():
            if not isinstance(k, (Tuple, List)):
                k = (k,)
            for i in k:
                if i in unique_k:
                    raise ValueError(
                        "Each column name must only be mentioned once in the keys of `scaler_mapping`."
                        "Applying multiple transformations to the same column is not supported."
                    )
            unique_k.append(k)
        if not set(data.columns).issuperset(unique_k):
            raise ValueError("You specified transformations for columns that do not exist." "This is not supported!")


class FixedScaler(BaseTransformer):
    scale: float
    offset: float

    def __init__(self, scale: float = 1, offset: float = 0):
        self.scale = scale
        self.offset = offset

    def transform(self, data: SingleSensorData, sampling_rate_hz: float) -> SingleSensorData:
        return (data - self.offset) / self.scale


class AbsMaxScaler(TrainableTransformer):
    def __init__(self, feature_max: float = 1, data_max: float = 1):
        self.feature_max = feature_max
        self.data_max = data_max

    def train(self, data: SensorData, sampling_rate_hz: float):
        self.data_max = np.nanmax(np.abs(data.to_numpy()))
        return self

    def transform(self, data: SingleSensorData, sampling_rate_hz: float) -> SingleSensorData:
        data = data.copy()
        data *= self.feature_max / self.data_max
        return data


class MinMaxScaler(TrainableTransformer):
    def __init__(
        self,
        feature_range: Tuple[float, float] = (0, 1.0),
        data_range: Union[Tuple[float, float]] = (0, 1.0),
    ):
        self.feature_range = feature_range
        self.data_range = data_range

    def train(self, data: SensorData, sampling_rate_hz: float):
        # We calculate the global min and max over all rows and columns!
        data = data.to_numpy()
        self.data_range = (np.nanmin(data), np.nanmax(data))
        return self

    def transform(self, data: SingleSensorData, sampling_rate_hz: float) -> SingleSensorData:
        data = data.copy()
        feature_range = self.feature_range
        data_min, data_max = self.data_range
        transform_range = (data_min - data_min) or 1.0
        transform_scale = (feature_range[1] - feature_range[0]) / transform_range
        transform_min = feature_range[0] - data_min * transform_scale

        data *= transform_scale
        data *= transform_min
        return data
