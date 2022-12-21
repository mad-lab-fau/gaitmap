"""A Basic ZUPT detector based on moving windows on the norm."""
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from typing_extensions import Literal, Self

from gaitmap.base import BaseZuptDetector
from gaitmap.utils.array_handling import bool_array_to_start_end_array
from gaitmap.utils.consts import SF_ACC, SF_GYR
from gaitmap.utils.datatype_helper import SingleSensorData, is_single_sensor_data
from gaitmap.utils.exceptions import ValidationError
from gaitmap.utils.static_moment_detection import METRIC_FUNCTION_NAMES, find_static_samples, find_static_samples_shoe

SENSOR_NAMES = Literal["acc", "gyr"]  # noqa: invalid-name


def _validate_window(window_length_s: float, window_overlap: float, sampling_rate_hz: float) -> Tuple[int, int]:
    """Validate window_length and overlap."""
    window_length = round(sampling_rate_hz * window_length_s)
    if window_length < 3:
        raise ValidationError(
            f"The effective window size is smaller than 3 samples (`sampling_rate_hz`={sampling_rate_hz}, "
            f"`window_length_s`={window_length_s}). "
            "Specify a larger window length."
        )
    if not 0 <= window_overlap < 1:
        raise ValidationError("`window_overlap` must be `0 <= window_overlap < 1`")

    window_overlap_samples = round(window_length * window_overlap)
    if window_overlap_samples == window_length:
        raise ValidationError(
            "The effective window overlap after rounding is 1, because the window is to short. "
            "Either choose a smaller overlap or a larger window."
        )
    return window_length, window_overlap_samples


class _PerSampleDetectorMixin:
    per_sample_zupts_: np.ndarray

    @property
    def zupts_(self) -> pd.DataFrame:
        """Get the start and end values of all zupts."""
        start_ends = bool_array_to_start_end_array(self.per_sample_zupts_)
        if len(start_ends) > 0:
            return pd.DataFrame(start_ends, columns=["start", "end"])
        return pd.DataFrame(columns=["start", "end"])


class NormZuptDetector(BaseZuptDetector, _PerSampleDetectorMixin):
    """Detect ZUPTs based on either the Acc or the Gyro norm.

    The ZUPT method uses a sliding window approach with overlap.
    For overlapping windows the results will be combined with a logical "or" in the region of the overlap.
    I.e. if one of the overlapping windows is considered static, the entire region is.
    Neighboring static windows will be combined to a static region.

    All calculations are always performed on the norm of the selected sensor.

    At the moment only single sensor data is supported.

    ... note::
        Using the gyro signal with the metric "squared_mean" and a window overlap of len(window) - 1 samples, this ZUPT
        detector is equivalent to the ARED Zupt detector presented in [1]_.

    .. warning::
        Due to edge cases at the end of the input data where window size and overlap might not fit your data, the last
        window might be discarded for analysis and will therefore always be considered as non-static!

    Parameters
    ----------
    sensor
        The sensor (either `acc` or `gyr`) to apply the detection to.
        The detector will calculate the norm for the respective sensor.
    window_length_s : int
        Length of desired window in seconds.
        The real window length is calculated as `window_length_samples = round(sampling_rate_hz * window_length_s)`
    window_overlap : float
        The overlap between two neighboring windows as a fraction of the window length.
        Must be `0 <= window_overlap < 1`.
        Note that the window length is first converted into samples, before the effective overlap is calculated.
        The overlap in samples is calculated as `overlap_samples = round(window_length_samples * window_overlap)`
    inactive_signal_threshold : float
       Threshold to decide whether a window should be considered as active or inactive.
       Window will be tested on `metric(norm_window) <= threshold`.
       The unit of this value depends on the selected sensor and metric.
    metric : str
        Metric which will be calculated per window on the norm of the signal, one of the following strings

        'mean' (default)
            Calculates mean value per window
        'squared_mean'
            The same as mean, but the norm of the signal is squared before calculating the mean.
        'maximum'
            Calculates maximum value per window
        'median'
            Calculates median value per window
        'variance'
            Calculates variance value per window

    Other Parameters
    ----------------
    data
        The data passed to the detect method
    sampling_rate_hz
        The sampling rate of this data

    Attributes
    ----------
    zupts_
        A dataframe with the columns `start` and `end` specifying the start and end of all static regions in samples
    per_sample_zupts_
        A bool array with length `len(data)`.
        If the value is `True` for a sample, it is part of a static region.
    window_length_samples_
        The internally calculated window length in samples.
        This might be helpful for debugging
    window_overlap_samples_
        The internally calculated window overlap in samples.
        This might be helpful for debugging.

    References
    ----------
    .. [1] I. Skog, J.-O. Nilsson, P. Händel, and J. Rantakokko, “Zero-velocity detection—An algorithm evaluation,”
       IEEE Trans. Biomed. Eng., vol. 57, no. 11, pp. 2657–2666, Nov. 2010.

    See Also
    --------
    gaitmap.utils.array_handling.sliding_window_view: Details on the used windowing function for this method.

    """

    sensor: SENSOR_NAMES
    window_length_s: float
    window_overlap: Optional[float]
    metric: METRIC_FUNCTION_NAMES
    inactive_signal_threshold: float

    data: SingleSensorData
    sampling_rate_hz: float

    window_length_samples_: int
    window_overlap_samples_: int

    def __init__(
        self,
        *,
        sensor: SENSOR_NAMES = "gyr",
        window_length_s: float = 0.15,
        window_overlap: float = 0.5,
        metric: METRIC_FUNCTION_NAMES = "mean",
        inactive_signal_threshold: float = 15,
    ):
        self.sensor = sensor
        self.window_length_s = window_length_s
        self.window_overlap = window_overlap
        self.metric = metric
        self.inactive_signal_threshold = inactive_signal_threshold

    def detect(self: Self, data: SingleSensorData, sampling_rate_hz: float, **kwargs) -> Self:
        """Detect all ZUPT regions in the data.

        Parameters
        ----------
        data
            The data set holding the imu raw data
        sampling_rate_hz
            The sampling rate of the data

        Returns
        -------
        self
            The class instance with all result attributes populated

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        is_single_sensor_data(
            self.data, frame="any", check_acc=self.sensor == "acc", check_gyr=self.sensor == "gyr", raise_exception=True
        )

        self.window_length_samples_, self.window_overlap_samples_ = _validate_window(
            self.window_length_s, self.window_overlap, self.sampling_rate_hz
        )

        self.per_sample_zupts_ = find_static_samples(
            data.filter(like=self.sensor).to_numpy(),
            window_length=self.window_length_samples_,
            inactive_signal_th=self.inactive_signal_threshold,
            metric=self.metric,
            overlap=self.window_overlap_samples_,
        )

        return self


class ShoeZuptDetector(BaseZuptDetector, _PerSampleDetectorMixin):
    """Detect ZUPTSs using the SHOE algorithm.

    This is based on the papers [1]_ and [2]_ and uses as weighted sum of the gravity corrected acc and the gyro norm to
    detect the static moments.

    .. note:: The threshold is heavily dependent on the provided noise levels of the sensors.
              These values should theoretically be provided in the datasheet of the sensors, but most of the times, it
              makes sense to grid search the value, as the effective noise level might vary quite a bit.
              Note, that primarily the relation between the acc and the gyro noise is relevant, as it defines the
              weighting of the two signals in the ZUPT detection.
              Default values likely need to be adapted to your specific use case.

    Parameters
    ----------
    acc_noise_variance
        The variance of the noise of the accelerometer.
    gyr_noise_variance
        The variance of the noise of the gyroscope.
    inactive_signal_threshold
        Threshold to decide whether a window should be considered as active or inactive.
        This value heavily depends on the noise levels selected for the other parameters.
    window_length_s : int
        Length of desired window in seconds.
        The real window length is calculated as `window_length_samples = round(sampling_rate_hz * window_length_s)`
    window_overlap : float
        The overlap between two neighboring windows as a fraction of the window length.
        Must be `0 <= window_overlap < 1`.
        Note that the window length is first converted into samples, before the effective overlap is calculated.
        The overlap in samples is calculated as `overlap_samples = round(window_length_samples * window_overlap)`

    Other Parameters
    ----------------
    data
        The data passed to the detect method
    sampling_rate_hz
        The sampling rate of this data

    Attributes
    ----------
    zupts_
        A dataframe with the columns `start` and `end` specifying the start and end of all static regions in samples
    per_sample_zupts_
        A bool array with length `len(data)`.
        If the value is `True` for a sample, it is part of a static region.
    window_length_samples_
        The internally calculated window length in samples.
        This might be helpful for debugging
    window_overlap_samples_
        The internally calculated window overlap in samples.
        This might be helpful for debugging.


    """

    window_length_s: float
    window_overlap: Optional[float]
    inactive_signal_threshold: float
    acc_noise_variance: float
    gyr_noise_variance: float

    data: SingleSensorData
    sampling_rate_hz: float

    window_length_samples_: int
    window_overlap_samples_: int

    def __init__(
        self,
        *,
        acc_noise_variance: float = 10e-8,
        gyr_noise_variance: float = 10e-9,
        window_length_s: float = 0.15,
        window_overlap: float = 0.5,
        inactive_signal_threshold: float = 15,
    ):
        self.acc_noise_variance = acc_noise_variance
        self.gyr_noise_variance = gyr_noise_variance
        self.window_length_s = window_length_s
        self.window_overlap = window_overlap
        self.inactive_signal_threshold = inactive_signal_threshold

    def detect(self: Self, data: SingleSensorData, sampling_rate_hz: float, **kwargs) -> Self:
        """Detect all ZUPT regions in the data.

        Parameters
        ----------
        data
            The data set holding the imu raw data
        sampling_rate_hz
            The sampling rate of the data

        Returns
        -------
        self
            The class instance with all result attributes populated

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        is_single_sensor_data(self.data, frame="any", raise_exception=True)

        self.window_length_samples_, self.window_overlap_samples_ = _validate_window(
            self.window_length_s, self.window_overlap, self.sampling_rate_hz
        )

        self.per_sample_zupts_ = find_static_samples_shoe(
            gyr=data.filter(like="gyr").to_numpy(),
            acc=data.filter(like="acc").to_numpy(),
            acc_noise_var=self.acc_noise_variance,
            gyr_noise_var=self.gyr_noise_variance,
            window_length=self.window_length_samples_,
            overlap=self.window_overlap_samples_,
            inactive_signal_th=self.inactive_signal_threshold,
        )

        return self
