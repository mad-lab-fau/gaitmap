"""A Basic ZUPT detector based on moving windows on the norm."""
from typing import Optional, TypeVar

import numpy as np
import pandas as pd
from typing_extensions import Literal

from gaitmap.base import BaseZuptDetector
from gaitmap.utils.array_handling import bool_array_to_start_end_array
from gaitmap.utils.datatype_helper import SingleSensorData, is_single_sensor_data
from gaitmap.utils.static_moment_detection import METRIC_FUNCTION_NAMES, find_static_samples

Self = TypeVar("Self", bound="NormZuptDetector")

SENSOR_NAMES = Literal["acc", "gyr"]


class NormZuptDetector(BaseZuptDetector):
    """Detect ZUPTs based on either the Acc or the Gyro norm.

    The ZUPT method uses a sliding window approach with overlap.
    Overlapping windows will be combined with a logical "or" in the region of the the overlap.
    I.e. if one of the overlapping windows is considered static, the entire region is.
    Neighboring static windows will be combined to a static region.

    All calculations are always performed on the norm of the selected sensor.

    At the moment only single sensor data is supported.

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
       Window will be tested on `metric(norm_window) <= threshold`
    metric : str
        Metric which will be calculated per window on the norm of the signal, one of the following strings

        'mean' (default)
            Calculates mean value per window
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

    See Also
    --------
    gaitmap.utils.array_handling.sliding_window_view: Details on the used windowing function for this method.
    gaitmap.utils.array_handling.sliding_window_view: Details on the used windowing function for this method.

    """

    sensor: SENSOR_NAMES
    window_length_s: float
    window_overlap: Optional[float]
    metric: METRIC_FUNCTION_NAMES
    inactive_signal_threshold: float

    data: SingleSensorData
    sampling_rate_hz: float

    per_sample_zupts_: np.ndarray

    def __init__(
        self,
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

    @property
    def zupts_(self) -> pd.DataFrame:
        """Get the start and end values of all zupts."""
        return pd.DataFrame(bool_array_to_start_end_array(self.per_sample_zupts_), columns=["start", "end"])

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

        window_length = round(sampling_rate_hz * self.window_length_s)
        if window_length < 2:
            raise ValueError(
                f"The effective window size is smaller than 2 samples (`sampling_rate_hz`={sampling_rate_hz}, "
                f"`window_length_s`={self.window_length_s}). "
                "Specify a larger window length."
            )
        if not 0 <= self.window_overlap < 1:
            raise ValueError("`window_overlap` must be `0 <= window_overlap < 1`")

        window_overlap_samples = round(window_length * self.window_overlap)
        if window_overlap_samples == window_length:
            raise ValueError(
                "The effective window overlap after rounding is 1, because the window is to short. "
                "Either choose a smaller overlap or a larger window."
            )

        zupts = find_static_samples(
            data.filter(like=self.sensor).to_numpy(),
            window_length=window_length,
            inactive_signal_th=self.inactive_signal_threshold,
            metric=self.metric,
            overlap=window_overlap_samples,
        )

        self.per_sample_zupts_ = zupts

        return self
