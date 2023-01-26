"""Transformers that scale data to certain data ranges."""
from typing import Optional, Sequence, Tuple

import numpy as np
from tpcp import OptimizableParameter, Parameter
from typing_extensions import Self

from gaitmap.data_transform._base import BaseTransformer, TrainableTransformerMixin
from gaitmap.utils.datatype_helper import SensorData, SingleSensorData, is_single_sensor_data


class FixedScaler(BaseTransformer):
    """Apply a fixed scaling and offset to the data.

    The transformed data y is calculated as:

    .. code-block::

        y = (x - offset) / scale

    Parameters
    ----------
    scale
        Downscaling factor of the data.
        The data is divided by this value
    offset
        The offset that should be subtracted from the data.

    Attributes
    ----------
    transformed_data_
        The transformed data.

    Other Parameters
    ----------------
    data
        The data passed to the transform method.

    """

    scale: Parameter[float]
    offset: Parameter[float]

    def __init__(self, scale: float = 1, offset: float = 0):
        self.scale = scale
        self.offset = offset

    def transform(self, data: SingleSensorData, **_) -> Self:
        """Scale the data.

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
        self.transformed_data_ = (data - self.offset) / self.scale
        return self


class StandardScaler(BaseTransformer):
    """Apply a standard scaling to the data.

    The transformed data y is calculated as:

    .. code-block::

        y = (x - x.mean()) / x.std(ddof)

    .. note:: Only a single mean and std are calculated over the entire data (i.e. not per column).

    Parameters
    ----------
    ddof
        The degree of freedom used in the standard deviation calculation.

    Attributes
    ----------
    transformed_data_
        The transformed data.

    Other Parameters
    ----------------
    data
        The data passed to the transform method.

    """

    ddof: Parameter[int] = 1

    def __init__(self, ddof: int = 1):
        self.ddof = ddof

    def transform(self, data: SingleSensorData, **_) -> Self:
        """Scale the data.

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
        self.transformed_data_ = (data - data.to_numpy().mean()) / data.to_numpy().std(ddof=self.ddof)
        return self

    def _transform_data(self, data: SingleSensorData, mean, std) -> SingleSensorData:
        return (data - mean) / std


class TrainableStandardScaler(StandardScaler, TrainableTransformerMixin):
    """Apply a standard scaling to the data.

    The transformed data y is calculated as:

    Parameters
    ----------
    mean
        The mean of the training data.
        The value can either be set manually or automatically calculated from the training data using `self_optimize`.
    std
        The standard deviation of the training data.
        The value can either be set manually or automatically calculated from the training data using `self_optimize`.
    ddof
        The degree of freedom used in the standard deviation calculation.

    Attributes
    ----------
    transformed_data_
        The transformed data.

    Other Parameters
    ----------------
    data
        The data passed to the transform method.

    """

    mean: OptimizableParameter[Optional[float]]
    std: OptimizableParameter[Optional[float]]

    def __init__(self, mean: Optional[float] = None, std: Optional[float] = None, ddof: int = 1):
        self.mean = mean
        self.std = std
        super().__init__(ddof=ddof)

    def self_optimize(self, data: Sequence[SingleSensorData], **_) -> Self:
        # Iteratively calculate the overall mean and std using a two-pass algorithm
        # First pass: Calculate the mean:
        sum_vals = 0
        count = 0
        for dp in data:
            is_single_sensor_data(dp, check_gyr=False, check_acc=False, raise_exception=True)
            sum_vals += dp.to_numpy().sum()
            count += dp.to_numpy().size

        # Second pass: Calculate the std:
        mean = sum_vals / count
        sum_vals = 0
        for dp in data:
            sum_vals += ((dp.to_numpy() - mean) ** 2).sum()
        std = np.sqrt(sum_vals / (count - self.ddof))

        self.mean = mean
        self.std = std
        return self

    def transform(self, data: SingleSensorData, **_) -> Self:
        """Scale the data.

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
        if self.mean is None or self.std is None:
            raise ValueError(
                "The mean and std must be set before the data can be transformed. Use `self_optimize` to "
                "learn them from a trainingssequence."
            )
        self.transformed_data_ = self._transform_data(data, self.mean, self.std)
        return self


class AbsMaxScaler(BaseTransformer):
    """Scale data by its absolute maximum.

    The data y after the transform is calculated as

    .. code-block::

        y = x * out_max / max(abs(x))

    Note that the maximum over **all** columns is calculated.
    I.e. Only a single global scaling factor is applied to all the columns.

    Parameters
    ----------
    out_max
        The value the maximum will be scaled to.
        After scaling the absolute maximum in the data will be equal to this value.
        Note that if the absolute maximum corresponds to a minimum in the data, this minimum will be scaled to
        `-out_max`.

    Attributes
    ----------
    transformed_data_
        The transformed data.

    Other Parameters
    ----------------
    data
        The data passed to the transform method.

    """

    out_max: Parameter[float]

    def __init__(self, out_max: float = 1):
        self.out_max = out_max

    def transform(self, data: SingleSensorData, **_) -> Self:
        """Scale the data.

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
        self.transformed_data_ = self._transform(data, self._get_abs_max(data))
        return self

    def _get_abs_max(self, data: SingleSensorData) -> float:
        is_single_sensor_data(data, check_gyr=False, check_acc=False, raise_exception=True)
        return float(np.nanmax(np.abs(data.to_numpy())))

    def _transform(self, data: SingleSensorData, absmax: float) -> SingleSensorData:
        data = data.copy()
        data *= self.out_max / absmax
        return data


class TrainableAbsMaxScaler(AbsMaxScaler, TrainableTransformerMixin):
    """Scale data by the absolut max of a trainings sequence.

    .. warning :: By default, this scaler will not modify the data!
                  Use `self_optimize` to adapt the `data_max` parameter based on a set of training data.

    During training the scaler will calculate the absolute max from the trainings data,
    Per provided dataset `data_max` will be calculated.
    The final `data_max` is the max over all train sequences.

    .. code-block::

        data_max = max(abs(x_train))

    Note that the maximum over **all** columns is calculated.
    I.e. Only a single global scaling factor is applied to all the data.

    During transformation, this fixed scaling factor is applied to any new columns.

    .. code-block::

        y = x * out_max / data_max

    Parameters
    ----------
    out_max
        The value the maximum will be scaled to.
        After scaling the absolute maximum in the data will be equal to this value.
        Note that if the absolute maximum corresponds to a minimum in the data, this minimum will be scaled to
        `-feature_max`.
    data_max
        The maximum of the training data.
        The value can either be set manually or automatically calculated from the training data using `self_optimize`.

    Attributes
    ----------
    transformed_data_
        The transformed data.

    Other Parameters
    ----------------
    data
        The data passed to the transform method.

    """

    data_max: OptimizableParameter[Optional[float]]

    def __init__(self, out_max: float = 1, data_max: Optional[float] = None):
        self.data_max = data_max
        super().__init__(out_max=out_max)

    def self_optimize(self, data: Sequence[SingleSensorData], **_) -> Self:
        """Calculate scaling parameters based on a trainings sequence.

        Parameters
        ----------
        data
           A sequence of dataframes, each representing single-sensor data.

        Returns
        -------
        self
            The trained instance of the transformer

        """
        max_vals = [self._get_abs_max(d) for d in data]
        self.data_max = np.max(max_vals)
        return self

    def transform(self, data: SingleSensorData, **_) -> Self:
        """Scale the data.

        Parameters
        ----------
        data
            A dataframe representing single sensor data.

        Returns
        -------
        self
            The instance of the transformer with the results attached

        """
        if self.data_max is None:
            raise ValueError("data_max not set. Use self_optimize to learn it based on a trainings sequence.")
        self.data = data
        self.transformed_data_ = self._transform(data, self.data_max)
        return self


class MinMaxScaler(BaseTransformer):
    """Scale the data by its Min-Max values.

    After the scaling the min of the data is equivalent ot `out_range[0]` and the max of the data is equivalent
    to `out_range[1]`.
    The output y is calculated as follows:

    .. code-block::

        scale = (out_range[1] - out_range[0]) / (x.min(), x.max())
        offset = out_range[0] - x.min() * transform_scale
        y = x * scale + offset

    Note that the minimum and maximum over **all** columns is calculated.
    I.e. Only a single global scaling factor is applied to all the columns.

    Parameters
    ----------
    out_range
        The range the data is scaled to.

    Attributes
    ----------
    transformed_data_
        The transformed data.

    Other Parameters
    ----------------
    data
        The data passed to the transform method.

    """

    out_range: Parameter[Tuple[float, float]]

    def __init__(
        self,
        out_range: Tuple[float, float] = (0, 1.0),
    ):
        self.out_range = out_range

    def transform(self, data: SingleSensorData, **_) -> Self:
        """Scale the data.

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
        # Test if out data range is valid
        if self.out_range[0] >= self.out_range[1]:
            raise ValueError("out_range[0] (new min) must be smaller than out_range[1] (new max)")
        data_range = self._calc_data_range(data)
        self.transformed_data_ = self._transform(data, data_range)
        return self

    def _calc_data_range(self, data: SensorData) -> Tuple[float, float]:
        is_single_sensor_data(data, check_gyr=False, check_acc=False, raise_exception=True)
        # We calculate the global min and max over all rows and columns!
        data = data.to_numpy()
        return float(np.nanmin(data)), float(np.nanmax(data))

    def _transform(self, data: SingleSensorData, data_range: Tuple[float, float]) -> SingleSensorData:
        data = data.copy()
        feature_range = self.out_range
        data_min, data_max = data_range
        transform_range = (data_max - data_min) or 1.0
        transform_scale = (feature_range[1] - feature_range[0]) / transform_range
        transform_min = feature_range[0] - data_min * transform_scale

        data *= transform_scale
        data += transform_min
        return data


class TrainableMinMaxScaler(MinMaxScaler, TrainableTransformerMixin):
    """Scale the data by Min-Max values learned from trainings data.

    .. warning :: By default, this scaler will not modify the data!
                  Use `self_optimize` to adapt the `data_range` parameter based on a set of training data.

    During training the scaling and offset is calculated based on the min and max of the trainings sequence.
    If multiple sequences are provided for training, the global min and max values of **all** sequences are used.

    .. code-block::

        data_range =  (x_train.min(), x_train.max())
        scale = (out_range[1] - out_range[0]) / (data_range[1] - data_range[0])
        offset = out_range[0] - x_train.min() * transform_scale

    Note that the minimum and maximum over **all** columns is calculated.
    I.e. Only a single global scaling factor is applied to all the columns.

    During `transform` these trained transformation are applied as follows.

    .. code-block::

        y = x * scale + offset

    Parameters
    ----------
    out_range
        The range the data is scaled to.
    data_range
        The range of the data used for training.
        The values can either be set manually or automatically calculated from the training data using `self_optimize`.

    Attributes
    ----------
    transformed_data_
        The transformed data.

    Other Parameters
    ----------------
    data
        The data passed to the transform method.

    """

    data_range: OptimizableParameter[Optional[Tuple[float, float]]]

    def __init__(
        self,
        out_range: Tuple[float, float] = (0, 1.0),
        data_range: Optional[Tuple[float, float]] = None,
    ):
        self.data_range = data_range
        super().__init__(out_range=out_range)

    def self_optimize(self, data: Sequence[SingleSensorData], **_):
        """Calculate scaling parameters based on a trainings sequence.

        Parameters
        ----------
        data
           A sequence of dataframes, each representing single-sensor data.

        Returns
        -------
        self
            The trained instance of the transformer

        """
        mins, maxs = zip(*(self._calc_data_range(d) for d in data))
        self.data_range = np.min(mins), np.max(maxs)
        return self

    def transform(self, data: SingleSensorData, **_) -> Self:
        """Scale the data.

        Parameters
        ----------
        data
            A dataframe representing single sensor data.

        Returns
        -------
        self
            The instance of the transformer with the results attached

        """
        if self.data_range is None:
            raise ValueError("No data range set. Use self_optimize to learn it based on a trainings sequence.")
        self.data = data
        self.transformed_data_ = self._transform(data, data_range=self.data_range)
        return self
