from typing import Dict, Union, Tuple, List

import numpy as np
import pandas as pd

from gaitmap.base import _BaseSerializable
from gaitmap.utils._types import _Hashable
from gaitmap.utils.datatype_helper import SensorData, SingleSensorData


class BaseTransformer(_BaseSerializable):
    """Base class for all data transformers."""

    def transform(self, data: SingleSensorData, sampling_rate_hz: float) -> SingleSensorData:
        """Transform the data using the transformer.

        Parameters
        ----------
        data
            A dataframe representing single sensor data.
        sampling_rate_hz
            The sampling rate of the data in Hz

        Returns
        -------
        transformed_data
            The transformed data

        """
        raise NotImplementedError()


class TrainableTransformer(BaseTransformer):
    """Base class for transformers with adaptable parameters."""

    def train(self, data: SingleSensorData, sampling_rate_hz: float):
        """Learn the parameters of the transformer based on provided data.

        Parameters
        ----------
        data
           A dataframe representing single sensor data.
           # TODO: Does this make sense to only have a single sensor dataframe here?
        sampling_rate_hz
            The sampling rate of the data in Hz

        Returns
        -------
        self
            The trained instance of the transformer
        """
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
    """Apply a fixed scaling and offset to the data.

    The transformed data y is calculated as:

    `y = (x - offset) / scale`

    Parameters
    ----------
    scale
        Downscaling factor of the data.
        The data is divided by this value
    offset
        The offset that should be subtracted from the data.

    """

    scale: float
    offset: float

    def __init__(self, scale: float = 1, offset: float = 0):
        self.scale = scale
        self.offset = offset

    def transform(self, data: SingleSensorData, sampling_rate_hz: float) -> SingleSensorData:
        """Scale the data.

        Parameters
        ----------
        data
            A dataframe representing single sensor data.
        sampling_rate_hz
            The sampling rate of the data in Hz

        Returns
        -------
        transformed_data
            The scaled dataframe

        """
        return (data - self.offset) / self.scale


class AbsMaxScaler(BaseTransformer):
    """Scale data by its absolute maximum.

    The data y after the transform is calculated as

    .. code-block::

        y = x * feature_max / max(abs(x))

    Note that the maximum over **all** columns is calculated.
    I.e. Only a single global scaling factor is applied to all the columns.

    Parameters
    ----------
    feature_max
        The value the maximum will be scaled to.
        After scaling the absolute maximum in the data will be equal to this value.
        Note that if the absolute maximum corresponds to a minimum in the data, this minimum will be scaled to
        `-feature_max`.

    """

    feature_max: float

    def __init__(self, feature_max: float = 1):
        self.feature_max = feature_max

    def transform(self, data: SingleSensorData, sampling_rate_hz: float) -> SingleSensorData:
        """Scale the data.

        Parameters
        ----------
        data
            A dataframe representing single sensor data.
        sampling_rate_hz
            The sampling rate of the data in Hz

        Returns
        -------
        transformed_data
            The scaled dataframe

        """
        return self._transform(data, self._get_abs_max(data))

    def _get_abs_max(self, data: SingleSensorData) -> float:
        return np.nanmax(np.abs(data.to_numpy()))

    def _transform(self, data: SingleSensorData, absmax: float) -> SingleSensorData:
        data = data.copy()
        data *= self.feature_max / absmax
        return data


class TrainableAbsMaxScaler(AbsMaxScaler, TrainableTransformer):
    """Scale data by the absolut max of a trainings sequence.

    .. warning :: By default, this scaler will not modify the data!
                  Use `train` to adapt the `data_range` parameter based on a set of training data.

    During training the scaler will calculate the absolute max from the trainigs data:

    .. code-block::

        data_max = max(abs(x_train))

    Note that the maximum over **all** columns is calculated.
    I.e. Only a single global scaling factor is applied to all the data.

    During transformation, this fixed scaling factor is applied to any new columns.

    .. code-block::

        y = x / data_max

    """

    def __init__(self, feature_max: float = 1, data_max: float = 1):
        self.data_max = data_max
        super().__init__(feature_max=feature_max)

    def train(self, data: SensorData, sampling_rate_hz: float):
        """Calculate scaling parameters based on a trainings sequence.

        Parameters
        ----------
        data
           A dataframe representing single sensor data.
        sampling_rate_hz
            The sampling rate of the data in Hz

        Returns
        -------
        self
            The trained instance of the transformer
        """
        self.data_max = self._get_abs_max(data)
        return self

    def transform(self, data: SingleSensorData, sampling_rate_hz: float) -> SingleSensorData:
        """Scale the data.

        Parameters
        ----------
        data
            A dataframe representing single sensor data.
        sampling_rate_hz
            The sampling rate of the data in Hz

        Returns
        -------
        transformed_data
            The scaled dataframe

        """
        return self._transform(data, self.data_max)


class MinMaxScaler(BaseTransformer):
    """Scale the data by Min-Max values learned from trainings data

    After the scaling the min of the data is equivalent ot `feature_range[0]` and the max of the data is equivalent
    to `feature_range[1]`.
    The output y is calculated as follows:

    .. code-block::

        scale = (feature_range[1] - feature_range[0]) / (x.min(), x.max())
        offset = feature_range[0] - x.min() * transform_scale
        y = x * scale + offset

    Note that the minimum and maximum over **all** columns is calculated.
    I.e. Only a single global scaling factor is applied to all the columns.

    """

    def __init__(
        self,
        feature_range: Tuple[float, float] = (0, 1.0),
    ):
        self.feature_range = feature_range

    def transform(self, data: SingleSensorData, sampling_rate_hz: float) -> SingleSensorData:
        """Scale the data.

        Parameters
        ----------
        data
            A dataframe representing single sensor data.
        sampling_rate_hz
            The sampling rate of the data in Hz

        Returns
        -------
        transformed_data
            The scaled dataframe

        """
        data_range = self._calc_data_range(data)
        return self._transform(data, data_range)

    def _calc_data_range(self, data: SensorData) -> Tuple[float, float]:
        # We calculate the global min and max over all rows and columns!
        data = data.to_numpy()
        return np.nanmin(data), np.nanmax(data)

    def _transform(self, data: SingleSensorData, data_range) -> SingleSensorData:
        data = data.copy()
        feature_range = self.feature_range
        data_min, data_max = data_range
        transform_range = (data_min - data_min) or 1.0
        transform_scale = (feature_range[1] - feature_range[0]) / transform_range
        transform_min = feature_range[0] - data_min * transform_scale

        data *= transform_scale
        data += transform_min
        return data


class TrainableMinMaxScaler(MinMaxScaler, TrainableTransformer):
    """Scale the data by Min-Max values learned from trainings data

    .. warning :: By default, this scaler will not modify the data!
              Use `train` to adapt the `data_range` parameter based on a set of training data.

    During training the scaling and offset is calculated based on the min and max of the trainings sequence:

    .. code-block::
        data_range =  (x_train.min(), x_train.max())
        scale = (feature_range[1] - feature_range[0]) / (data_range[1] - data_range[0])
        offset = feature_range[0] - x_train.min() * transform_scale

    Note that the minimum and maximum over **all** columns is calculated.
    I.e. Only a single global scaling factor is applied to all the columns.

    During `transform` these trained transformation are applied as follows.

    .. code-block::

        y = x * scale + offset

    """

    def __init__(
        self,
        feature_range: Tuple[float, float] = (0, 1.0),
        data_range: Union[Tuple[float, float]] = (0, 1.0),
    ):
        self.data_range = data_range
        super().__init__(feature_range=feature_range)

    def train(self, data: SensorData, sampling_rate_hz: float):
        """Calculate scaling parameters based on a trainings sequence.

        Parameters
        ----------
        data
           A dataframe representing single sensor data.
        sampling_rate_hz
            The sampling rate of the data in Hz

        Returns
        -------
        self
            The trained instance of the transformer
        """
        self.data_range = self._calc_data_range(data)
        return self

    def transform(self, data: SingleSensorData, sampling_rate_hz: float) -> SingleSensorData:
        """Scale the data.

        Parameters
        ----------
        data
            A dataframe representing single sensor data.
        sampling_rate_hz
            The sampling rate of the data in Hz

        Returns
        -------
        transformed_data
            The scaled dataframe

        """
        return self._transform(data, self.data_range)
