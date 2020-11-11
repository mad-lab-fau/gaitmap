"""A set of signal processing functions."""

import numpy as np
from numba import njit
from scipy.signal import butter, lfilter


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
    np.array([0.00000000e+00, 4.82434336e-03, 4.03774045e-02, 1.66525148e-01,...])

    """
    b, a = butter(order, cutoff_freq_hz, btype="low", analog=False, fs=sampling_rate_hz)
    data_filtered = lfilter(b, a, data)
    return data_filtered


@njit(nogil=True, parallel=True, cache=True)
def row_wise_autocorrelation(array: np.ndarray, lag_max: int):
    """Compute the autocorrelation function row-wise for a 2d array.

    Parameters
    ----------
    array : array with shape (n,m)
        array which holds in every row a signal (window) for which the autocorrelation function should be computed
    lag_max : int
        the maximum lag for which the autocorrelation function should be computed. This relates to the lower
        frequency bound that is required.

    Returns
    -------
    A 2d array that holds the autocorrelation function for each row in the input array

    Examples
    --------
    >>> t = np.arange(0,1,0.1)
    >>> sin_wave = np.sin(t)
    >>> array = np.array([sin_wave, sin_wave])
    >>> out = row_wise_autocorrelation(array, 5)
    >>> out
    np.array([[2.38030226, 2.03883723, 1.68696752, 1.33807603, 1.00531772, 0.70139157],
       [2.38030226, 2.03883723, 1.68696752, 1.33807603, 1.00531772, 0.70139157]])

    """
    out = np.empty((array.shape[0], lag_max + 1))
    for tau in range(lag_max + 1):
        tmax = array.shape[1] - tau
        umax = array.shape[1] + tau
        out[:, tau] = (array[:, :tmax] * array[:, tau:umax]).sum(axis=1)
    return out
