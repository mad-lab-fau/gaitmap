"""A set of transformers that can be used to calculate traditional features from a timeseries."""
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
from gaitmap.utils.datatype_helper import SingleSensorData, SingleSensorRegionsOfInterestList


class Resample(BaseTransformer):
    """Resample a time series using the scipy resample method.

    Optionally this method can also convert a ROI list (with start end values) into the same new sampling rate so
    that it still matches the resampled data.

    Parameters
    ----------
    target_sampling_rate_hz
        The target sampling rate the data should be resampled to.
        Note that we don't apply any checks on that.
        If you upsample your data to far, you will likely get poor results.

    Attributes
    ----------
    transformed_data_
        The transformed data.
    transformed_roi_list_
        If a roi_list was provided, this will be the transformed roi list in the new sampling rate

    Other Parameters
    ----------------
    data
        The data passed to the transform method.
    roi_list
        Optional roi list (with values in samples) passed to the transform method
    sampling_rate_hz
        The sampling rate of the input data

    """

    target_sampling_rate_hz: Optional[float]

    sampling_rate_hz: float
    roi_list: SingleSensorRegionsOfInterestList

    transformed_roi_list_: SingleSensorRegionsOfInterestList

    def __init__(
        self,
        target_sampling_rate_hz: Optional[float] = None,
    ):
        self.target_sampling_rate_hz = target_sampling_rate_hz

    def transform(
        self,
        data: Optional[SingleSensorData] = None,
        *,
        roi_list: Optional[SingleSensorRegionsOfInterestList] = None,
        sampling_rate_hz: Optional[float] = None,
        **kwargs,
    ) -> Self:
        """Resample the data.

        Parameters
        ----------
        data
            data to be filtered
        roi_list
            Optional roi list (with values in samples), that will also be resampled to match the data at the new
            sampling rate.
            Note, that only the start and end columns will be modified.
            Other columns remain untouched.
        sampling_rate_hz
            The sampling rate of the data in Hz

        """
        if sampling_rate_hz is None:
            raise ValueError(f"{type(self).__name__}.transform requires a `sampling_rate_hz` to be passed.")
        if self.target_sampling_rate_hz is None:
            raise ValueError(
                f"{type(self).__name__} requires a `target_sampling_rate_hz` to be specified "
                "as parameter."
                "This can be done by passing it to the constructor or using the `set_params` method."
            )

        self.sampling_rate_hz = sampling_rate_hz

        if sampling_rate_hz == self.target_sampling_rate_hz:
            if data is not None:
                self.data = data
                self.transformed_data_ = copy(data)
            if roi_list is not None:
                self.roi_list = roi_list
                self.transformed_roi_list_ = copy(roi_list)
            return self

        if data is not None:
            self.data = data
            n_samples = int(np.round(len(data) * self.target_sampling_rate_hz / self.sampling_rate_hz))

            data_resampled = signal.resample(data, n_samples, axis=0)
            if isinstance(data, pd.DataFrame):
                data_resampled = pd.DataFrame(data_resampled, columns=data.columns)
            elif isinstance(data, pd.Series):
                data_resampled = pd.Series(data_resampled, name=data.name)

            self.transformed_data_ = data_resampled
        if roi_list is not None:
            self.roi_list = roi_list
            out = roi_list.copy()
            out.loc[:, ["start", "end"]] = (
                (roi_list[["start", "end"]] * self.target_sampling_rate_hz / self.sampling_rate_hz).round().astype(int)
            )
            self.transformed_roi_list_ = out
        return self


class BaseSlidingWindowFeatureTransform(BaseTransformer):
    """Baseclass for all Sliding window feature transforms."""

    window_size_s: Optional[float]

    sampling_rate_hz: float

    def __init__(self, window_size_s: Optional[float] = None):
        self.window_size_s = window_size_s

    @property
    def effective_window_size_samples_(self) -> int:
        """Get the real sample size of the window in samples after rounding effects."""
        win_size = int(np.round(self.sampling_rate_hz * self.window_size_s))
        if win_size % 2 == 0:
            # We always want to have an odd window size to make sure that we can get a centered view.
            win_size += 1
        return win_size

    def transform(self, data: SingleSensorData, *, sampling_rate_hz: Optional[float] = None, **kwargs) -> Self:
        """Apply the transformation on each sliding window.

        Parameters
        ----------
        data
            data to be filtered
        sampling_rate_hz
            The sampling rate of the data in Hz

        """
        if sampling_rate_hz is None:
            raise ValueError(f"{type(self).__name__}.transform requires a `sampling_rate_hz` to be passed.")

        if self.window_size_s is None:
            raise ValueError(
                "A `window_size_s` must be specified, before the transform can be performed. "
                "This can be done by passing it to the constructor or using the `set_params` method."
            )

        self.sampling_rate_hz = sampling_rate_hz
        self.data = data
        self.transformed_data_ = self._transform(data, sampling_rate_hz, **kwargs)
        return self

    def _transform(self, data: SingleSensorData, sampling_rate_hz: float, **kwargs) -> SingleSensorData:
        raise NotImplementedError()


class _PandasRollingFeatureTransform(BaseSlidingWindowFeatureTransform):
    """Baseclass for sliding window transforms, that directly use pandas rolling."""

    _rolling_method_name: str
    _prefix: str

    def _transform(self, data: SingleSensorData, sampling_rate_hz: float, **kwargs) -> SingleSensorData:
        rolling_result = self._apply_rolling(
            data.rolling(self.effective_window_size_samples_, min_periods=1, center=True)
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
    """Calculate a sliding window mean.

    Parameters
    ----------
    window_size_s
        The window size in seconds

    Attributes
    ----------
    transformed_data_
        The transformed data.
    effective_window_size_samples_
        The effective length window in samples after rounding effects are taking into account.

    Other Parameters
    ----------------
    data
        The data passed to the transform method.
    sampling_rate_hz
        The sampling rate of the input data

    """

    _rolling_method_name = "mean"
    _prefix = "mean"


class SlidingWindowVar(_PandasRollingFeatureTransform):
    """Calculate a sliding window variance.

    Parameters
    ----------
    window_size_s
        The window size in seconds

    Attributes
    ----------
    transformed_data_
        The transformed data.
    effective_window_size_samples_
        The effective length window in samples after rounding effects are taking into account.

    Other Parameters
    ----------------
    data
        The data passed to the transform method.
    sampling_rate_hz
        The sampling rate of the input data

    """

    _rolling_method_name = "var"
    _prefix = "var"


class SlidingWindowStd(_PandasRollingFeatureTransform):
    """Calculate a sliding window standard deviation.

    Parameters
    ----------
    window_size_s
        The window size in seconds

    Attributes
    ----------
    transformed_data_
        The transformed data.
    effective_window_size_samples_
        The effective length window in samples after rounding effects are taking into account.

    Other Parameters
    ----------------
    data
        The data passed to the transform method.
    sampling_rate_hz
        The sampling rate of the input data

    """

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

    return sliding_window_view(array, window_size_samples, window_size_samples - 1).swapaxes(0, 1)


class _CustomSlidingWindowTransform(BaseSlidingWindowFeatureTransform):
    def _apply_to_window_view(self, windowed_view: np.ndarray, data: pd.DataFrame):
        raise NotImplementedError

    def _transform(self, data: SingleSensorData, sampling_rate_hz: float, **kwargs) -> SingleSensorData:
        windowed_view = _get_centered_window_view(data.to_numpy(), self.effective_window_size_samples_, pad_value=0.0)
        # We pass the original data, so that the method can correctly create an output in form of a Dataframe
        return self._apply_to_window_view(windowed_view, data)


class SlidingWindowGradient(_CustomSlidingWindowTransform):
    """Calculate a sliding gradient by fitting a linear function on every sliding window.

    Parameters
    ----------
    window_size_s
        The window size in seconds

    Attributes
    ----------
    transformed_data_
        The transformed data.
    effective_window_size_samples_
        The effective length window in samples after rounding effects are taking into account.

    Other Parameters
    ----------------
    data
        The data passed to the transform method.
    sampling_rate_hz
        The sampling rate of the input data

    """

    def _apply_to_window_view(self, windowed_view: np.ndarray, data: pd.DataFrame):
        # To handle 2D and 3D input, we reshape 3D window views to 2D and then apply the 2D polyfit function.
        # This is done by flattening the last two dimensions.
        if windowed_view.ndim == 3:
            reshaped_windowed_view = windowed_view.reshape(windowed_view.shape[0], -1)
        else:
            reshaped_windowed_view = windowed_view

        # We only get the [0] element of the polyfit result, because we are only interested in the slope.
        reshaped_fit_coefs = polyfit(np.arange(self.effective_window_size_samples_), reshaped_windowed_view, 1)[0]

        if windowed_view.ndim == 3:
            fit_coefs = reshaped_fit_coefs.reshape(windowed_view.shape[1:])
        else:
            fit_coefs = reshaped_fit_coefs

        if isinstance(data, pd.Series):
            return pd.Series(fit_coefs, name=f"gradient__{data.name}")
        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(fit_coefs, columns=data.columns).rename(columns=lambda x: f"gradient__{x}")
        # We should never end up here...
        raise ValueError("Unexpected input dtype")
