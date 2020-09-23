"""A set of signal processing functions."""

from scipy.signal import butter, lfilter
import numpy as np


def butter_lowpass_filter_1d(data: np.ndarray, sampling_rate_hz: float, cutoff_freq_hz: float, order: int = 4):
    """Create and apply butterworth filter for a 1d signal.

    Parameters
    ----------
    data : array with shape (n,)
        array which holds the signal that is supposed to be filtered

    cutoff_freq_hz : float
        cutoff frequency for the lowpass filter

    sampling_rate_hz : float
        sampling rate of the signal in `data` supposed to be filtered

    order : int
        the order of the filter

    Returns
    -------
    the filtered signal

    Examples
    --------
    >>> data = np.arange(0,100)
    >>> data_filtered = butter_lowpass_filter_1d(data = data, sampling_rate_hz = 10, cutoff_freq_hz = 1,
    >>> order = 4)
    >>> data_filtered
    np.array([array([0.00000000e+00, 4.82434336e-03, 4.03774045e-02, 1.66525148e-01,...])

    """
    nyquist_frequency_hz = 0.5 * sampling_rate_hz
    normal_cutoff_freq = cutoff_freq_hz / nyquist_frequency_hz
    b, a = butter(order, normal_cutoff_freq, btype="low", analog=False)
    data_filtered = lfilter(b, a, data)
    return data_filtered
