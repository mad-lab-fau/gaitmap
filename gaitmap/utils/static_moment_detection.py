"""A set of util functions to detect static regions in a IMU signal given certain constrains."""
from functools import partial
from typing import Callable, Optional, Sequence, Tuple, get_args

import numpy as np
from numpy.linalg import norm
from typing_extensions import Literal

from gaitmap.utils import array_handling
from gaitmap.utils.array_handling import _bool_fill
from gaitmap.utils.consts import GRAV

# supported metric functions
_METRIC_FUNCTIONS = {"maximum": np.nanmax, "variance": np.nanvar, "mean": np.nanmean, "median": np.nanmedian}
METRIC_FUNCTION_NAMES = Literal["maximum", "variance", "mean", "median", "squared_mean"]  # noqa: invalid-name


def _window_apply_threshold(
    data, window_length: int, overlap: int, func: Callable[[np.ndarray], np.ndarray], threshold: float
):
    # allocate output array
    inactive_signal_bool_array = np.zeros(len(data))

    windowed_norm = np.atleast_2d(array_handling.sliding_window_view(data, window_length, overlap, nan_padding=False))
    is_static = np.broadcast_to(func(windowed_norm) <= threshold, windowed_norm.shape[::-1]).T

    # create the list of indices for sliding windows with overlap
    windowed_indices = np.atleast_2d(
        array_handling.sliding_window_view(np.arange(0, len(data)), window_length, overlap, nan_padding=False)
    )

    # iterate over sliding windows
    return _bool_fill(windowed_indices, is_static, inactive_signal_bool_array).astype(bool)


def find_static_samples(
    signal: np.ndarray,
    window_length: int,
    inactive_signal_th: float,
    metric: METRIC_FUNCTION_NAMES = "mean",
    overlap: Optional[int] = None,
) -> np.ndarray:
    """Search for static samples within given input signal, based on windowed L2-norm thresholding.

    .. warning::
        Due to edge cases at the end of the input data where window size and overlap might not fit your data, the last
        window might be discarded for analysis and will therefore always be considered as non-static!

    Parameters
    ----------
    signal : array with shape (n, 3)
        3D signal on which static moment detection should be performed (e.g. 3D-acc or 3D-gyr data)

    window_length : int
        Length of desired window in units of samples

    inactive_signal_th : float
       Threshold to decide whether a window should be considered as active or inactive. Window will be tested on
       <= threshold

    metric : str or Callable
        Metric which will be calculated per window, one of the following strings:

        'mean' (default)
            Calculates mean value per window
        'squared_mean'
            The same as mean, but the norm of the signal is squared before calculating the mean.
            This option exists to provide an implementation of the ARED Zupt detector [1]_.
        'maximum'
            Calculates maximum value per window
        'median'
            Calculates median value per window
        'variance'
            Calculates variance value per window

    overlap : int, optional
        Length of desired overlap in units of samples. If None (default) overlap will be window_length - 1

    Returns
    -------
    Boolean array with length n to indicate static (=True) or non-static (=False) for each sample

    Examples
    --------
    >>> test_data = load_gyro_data(path)
    >>> get_static_moments(gyro_data, window_length=128, overlap=64, inactive_signal_th = 5, metric = 'mean')

    References
    ----------
    .. [1] I. Skog, J.-O. Nilsson, P. Händel, and J. Rantakokko, “Zero-velocity detection—An algorithm evaluation,”
       IEEE Trans. Biomed. Eng., vol. 57, no. 11, pp. 2657–2666, Nov. 2010.

    See Also
    --------
    gaitmap.utils.array_handling.sliding_window_view: Details on the used windowing function for this method.

    """
    # test for correct input data shape
    if np.shape(signal)[-1] != 3:
        raise ValueError("Invalid signal dimensions, signal must be of shape (n,3).")

    if metric not in get_args(METRIC_FUNCTION_NAMES):
        raise ValueError("Invalid metric passed! {} as metric is not supported.".format(metric))

    # check if minimum signal length matches window length
    if window_length > len(signal):
        raise ValueError(
            "Invalid window length, window must be smaller or equal than given signal length. Given signal length: "
            "{} with given window_length: {}.".format(len(signal), window_length)
        )

    # add default overlap value
    if overlap is None:
        overlap = window_length - 1

    # calculate norm of input signal (do this outside of loop to boost performance at cost of memory!)
    if metric == "squared_mean":
        signal_norm = np.square(norm(signal, axis=1))
        metric = "mean"
    else:
        signal_norm = norm(signal, axis=1)

    mfunc = partial(_METRIC_FUNCTIONS[metric], axis=1)

    return _window_apply_threshold(signal_norm, window_length, overlap, mfunc, inactive_signal_th)


def find_static_samples_shoe(
    acc: np.ndarray,
    gyr: np.ndarray,
    acc_noise_var: float,
    gyr_noise_var: float,
    window_length: int,
    inactive_signal_th: float,
    overlap: Optional[int] = None,
) -> np.ndarray:
    """Use the SHOE algorithm for static moment detection.

    This is based on the papers [1]_ and [2]_ and uses as weighted sum of the gravity corrected acc and the gyro norm to
    detect the static moments.

    Parameters
    ----------
    acc : array with shape (n, 3)
        3D acc signal on which static moment detection should be performed
    gyr : array with shape (n, 3)
        3D gyr signal on which static moment detection should be performed
    acc_noise_var : float
        Variance of the noise in the acc sensor.
        This might be derived from the datasheet of the sensor, but usually it is a good idea to gridsearch this value.
    gyr_noise_var : float
        Variance of the noise in the gyr sensor.
        This might be derived from the datasheet of the sensor, but usually it is a good idea to gridsearch this value.
    window_length : int
        Length of desired window in units of samples
    inactive_signal_th : float
         Threshold to decide whether a window should be considered as active or inactive. Window will be tested on
            <= threshold
    overlap : int, optional
        Length of desired overlap in units of samples. If None (default) overlap will be window_length - 1

    References
    ----------
    .. [1] I. Skog, J.-O. Nilsson, P. Händel, and J. Rantakokko, “Zero-velocity detection—An algorithm evaluation,”
       IEEE Trans. Biomed. Eng., vol. 57, no. 11, pp. 2657–2666, Nov. 2010.
    .. [2] Wagstaff, Peretroukhin, and Kelly, “Robust Data-Driven Zero-Velocity Detection for Foot-Mounted Inertial
       Navigation.”

    """
    # check if minimum signal length matches window length
    assert acc.shape == gyr.shape, "Acc and Gyr data must have the same shape!"

    if window_length > len(acc):
        raise ValueError(
            "Invalid window length, window must be smaller or equal than given signal length. Given signal length: "
            f"{len(acc)} with given window_length: {window_length}."
        )

    # add default overlap value
    if overlap is None:
        overlap = window_length - 1

    # For acc we try to remove the influence of gravity first, by substracting the value of gravity in the direction
    # the average acceleration
    acc_mean = np.mean(acc, axis=0)
    grav_direction = acc_mean / norm(acc_mean)
    acc_norm = np.square(norm(acc - GRAV * grav_direction, axis=1))
    gyr_norm = np.square(norm(gyr, axis=1))
    combined = acc_norm / acc_noise_var + gyr_norm / gyr_noise_var

    return _window_apply_threshold(combined, window_length, overlap, partial(np.nanmean, axis=1), inactive_signal_th)


def find_static_sequences(
    signal: np.ndarray,
    window_length: int,
    inactive_signal_th: float,
    metric: METRIC_FUNCTION_NAMES = "mean",
    overlap: int = None,
) -> np.ndarray:
    """Search for static sequences within given input signal, based on windowed L2-norm thresholding.

    .. warning::
        Due to edge cases at the end of the input data where window size and overlap might not fit your data, the last
        window might be discarded for analysis and will therefore always be considered as non-static!

    Parameters
    ----------
    signal : array with shape (n, 3)
        3D signal on which static moment detection should be performed (e.g. 3D-acc or 3D-gyr data)

    window_length : int
        Length of desired window in units of samples

    inactive_signal_th : float
       Threshold to decide whether a window should be considered as active or inactive. Window will be tested on
       <= threshold

    metric : str, optional
        Metric which will be calculated per window, one of the following strings

        'mean' (default)
            Calculates mean value per window
        'maximum'
            Calculates maximum value per window
        'median'
            Calculates median value per window
        'variance'
            Calculates variance value per window

    overlap : int, optional
        Length of desired overlap in units of samples. If None (default) overlap will be window_length - 1

    Returns
    -------
    Array of [start, end] labels indication static regions within the input signal

    Examples
    --------
    >>> gyro_data = load_gyro_data(path)
    >>> static_regions = get_static_moment_labels(gyro_data, window_length=128, overlap=64, inactive_signal_th=5)

    See Also
    --------
    gaitmap.utils.array_handling.sliding_window_view: Details on the used windowing function for this method.

    """
    static_moment_bool_array = find_static_samples(
        signal=signal,
        window_length=window_length,
        inactive_signal_th=inactive_signal_th,
        metric=metric,
        overlap=overlap,
    )
    return array_handling.bool_array_to_start_end_array(static_moment_bool_array)


def find_first_static_window_multi_sensor(
    signals: Sequence[np.ndarray], window_length: int, inactive_signal_th: float, metric: METRIC_FUNCTION_NAMES
) -> Tuple[int, int]:
    """Find the first time window in the signal where all provided sensors are static.

    Parameters
    ----------
    signals : Sequence of n arrays with shape (k, m) or a 3D-array with shape (k, n, m)
        The signals of n senors with m axis and k samples.
    window_length
        Length of the required static signal in samples
    inactive_signal_th
        The threshold for static windows.
        If metric(norm(window, axis=-1))<=`inactive_signal_th` for all sensors, it is considered static.
    metric
        The metric that should be calculated on the vectornorm over all axis for each sensor in each window

    Returns
    -------
    (start, end)
        Start and end index of the first static window.

    Examples
    --------
    >>> sensor_1_gyro = ...
    >>> sensor_2_gyro = ...
    >>> find_first_static_window_multi_sensor([sensor_1_gyro, sensor_2_gyro], window_length=128, inactive_signal_th=5)

    """
    if metric not in _METRIC_FUNCTIONS:
        raise ValueError("`metric` must be one of {}".format(list(_METRIC_FUNCTIONS.keys())))

    if not isinstance(signals, np.ndarray):
        # all signals should have the same shape
        if not all(signals[0].shape == signal.shape for signal in signals):
            raise ValueError("All provided signals need to have the same shape.")
        if signals[0].ndim != 2:
            raise ValueError(
                "The array of each sensor must be 2D, where the first dimension is the time and the second dimension "
                "the sensor axis."
            )
        stacked_arrays = np.hstack(signals)
    else:
        if signals.ndim != 3:
            raise ValueError(
                "If a array is used as input, it must be 3D, where the first dimension is the time, "
                "the second indicates the sensor and the third the axis of the sensor."
            )
        stacked_arrays = signals

    n_signals = stacked_arrays.shape[1]

    windows = array_handling.sliding_window_view(
        stacked_arrays.reshape((stacked_arrays.shape[0], -1)),
        window_length=window_length,
        overlap=window_length - 1,
        nan_padding=False,
    )
    reshaped_windows = windows.reshape((*windows.shape[:-1], n_signals, -1))
    window_norm = norm(reshaped_windows, axis=-1)

    method = _METRIC_FUNCTIONS[metric]
    # This is pretty wasteful as we calculate the the function on all windows, even though we are only interested in
    # the first, where our threshold is valid.
    window_over_thres = method(window_norm, axis=1).max(axis=-1) <= inactive_signal_th

    valid_windows = np.nonzero(window_over_thres)[0]
    if len(valid_windows) == 0:
        raise ValueError("No static window was found")

    return valid_windows[0], valid_windows[0] + window_length
