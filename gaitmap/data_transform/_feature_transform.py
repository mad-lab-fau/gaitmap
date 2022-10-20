"""A set of transformers that can be used to calculate traditional features from a timeseries."""
import warnings
from copy import copy
from typing import Optional

import numpy as np
import pandas as pd
from numpy import polyfit
from pandas.core.window import Rolling
from scipy import signal
from typing_extensions import Self

from gaitmap.data_transform._base import BaseTransformer
from gaitmap.utils.array_handling import sliding_window_view
from gaitmap.utils.datatype_helper import SingleSensorData


class Resample(BaseTransformer):
    """Resample a time series using the scipy resample method."""

    target_sampling_rate_hz: float

    sampling_rate_hz: float

    def __init__(self, target_sampling_rate_hz: float, ):
        self.target_sampling_rate_hz = target_sampling_rate_hz

    @property
    def new_timeseries_(self):
        return


    def transform(self, data: SingleSensorData, *, sampling_rate_hz: Optional[float] = None, **kwargs) -> Self:
        if sampling_rate_hz is None:
            raise ValueError(f"{type(self).__name__}.transform requires a `sampling_rate_hz` to be passed.")

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        if sampling_rate_hz == self.target_sampling_rate_hz:
            self.transformed_data_ = copy(data)
            return self

        n_samples = int(np.round(len(data) * self.target_sampling_rate_hz / self.sampling_rate_hz))

        data_resampled = signal.resample(data, n_samples, axis=0)
        if isinstance(data, pd.DataFrame):
            data_resampled = pd.DataFrame(data_resampled, columns=data.columns)
        elif isinstance(data, pd.Series):
            data_resampled = pd.Series(data_resampled, name=data.name)

        self.transformed_data_ = data_resampled

        return self

    # def transform_roi_list(self, roi_list: Union[SingleSensorStrideList, SingleSensorRegionsOfInterestList]):


class BaseSlidingWindowFeatureTransform(BaseTransformer):
    """Baseclass for all Sliding window feature transforms."""

    window_size_s: Optional[float]

    sampling_rate_hz: float

    def __init__(self, window_size_s: Optional[float] = None):
        self.window_size_s = window_size_s

    @property
    def effective_window_size_samples_(self) -> int:
        win_size = int(np.round(self.sampling_rate_hz * self.window_size_s))
        if win_size % 2 == 0:
            # We always want to have an odd window size to make sure that we can get a centered view.
            win_size += 1
        return win_size

    def transform(self, data: SingleSensorData, *, sampling_rate_hz: Optional[float] = None, **kwargs) -> Self:
        if sampling_rate_hz is None:
            raise ValueError(f"{type(self).__name__}.transform requires a `sampling_rate_hz` to be passed.")

        if self.window_size_s is None:
            raise ValueError("A `window_size_s` must be specified, before the transform can be performed.")

        self.sampling_rate_hz = sampling_rate_hz
        self.data = data
        self.transformed_data_ = self._transform(data, sampling_rate_hz, **kwargs)
        return self

    def _transform(self, data: SingleSensorData, sampling_rate_hz: float, **kwargs) -> SingleSensorData:
        raise NotImplementedError()


class _PandasRollingFeatureTransform(BaseSlidingWindowFeatureTransform):
    _rolling_method_name: str
    _prefix: str

    def _transform(self, data: SingleSensorData, sampling_rate_hz: float, **kwargs) -> SingleSensorData:
        method = "table" if isinstance(data, pd.DataFrame) else "single"
        rolling_result = self._apply_rolling(
            data.rolling(self.effective_window_size_samples_, min_periods=1, center=True, method=method)
        )
        if isinstance(rolling_result, pd.DataFrame):
            return rolling_result.rename(columns=lambda x: f"{self._prefix}__{x}")
        if isinstance(rolling_result, pd.Series):
            return rolling_result.rename(f"{self._prefix}__{rolling_result.name}")
        # We should never end up here...
        raise ValueError("Unexpected return value from data.rolling")

    def _apply_rolling(self, rolling: Rolling) -> SingleSensorData:
        return getattr(rolling, self._rolling_method_name)()


class SlidingWindowMean(_PandasRollingFeatureTransform):
    _rolling_method_name = "mean"
    _prefix = "mean"


class SlidingWindowVar(_PandasRollingFeatureTransform):
    _rolling_method_name = "var"
    _prefix = "var"


class SlidingWindowStd(_PandasRollingFeatureTransform):
    _rolling_method_name = "std"
    _prefix = "std"


def _get_centered_window_view(array, window_size_samples, pad_value=0.0):
    pad_length = int(np.floor(window_size_samples / 2))

    if array.ndim == 1:
        array = np.pad(array.astype(float), (pad_length, pad_length), constant_values=pad_value)
    elif array.ndim == 2:
        array = np.pad(array.astype(float), [(pad_length, pad_length), (0, 0)], constant_values=pad_value)
    else:
        raise ValueError("Only error of dim 1 or 2 can be turned into sliding windows with this method.")

    return sliding_window_view(array, window_size_samples, window_size_samples - 1)


class _CustomSlidingWindowTransform(BaseSlidingWindowFeatureTransform):
    def _apply_to_window_view(self, windowed_view: np.ndarray, data: pd.DataFrame):
        raise NotImplementedError

    def _transform(self, data: SingleSensorData, sampling_rate_hz: float, **kwargs) -> SingleSensorData:
        windowed_view = _get_centered_window_view(data.to_numpy(), self.effective_window_size_samples_, pad_value=0.0)
        # We pass the original data, so that the method can correctly create an output in form of a Dataframe
        return self._apply_to_window_view(windowed_view, data)


class SlidingWindowGradient(_CustomSlidingWindowTransform):
    def _apply_to_window_view(self, windowed_view: np.ndarray, data: pd.DataFrame):
        # TODO: This only works with 1D input arrays -> 2D windowed views, but not 2D input arrays -> 3D windowed views.
        fit_coefs = polyfit(np.arange(self.effective_window_size_samples_), windowed_view.T, 1)[0]
        if isinstance(data, pd.Series):
            return pd.Series(fit_coefs, name=f"gradient__{data.name}")
        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(fit_coefs, columns=data.columns).rename(columns=lambda x: f"gradient__{x}")
        # We should never end up here...
        raise ValueError("Unexpected input dtype")
