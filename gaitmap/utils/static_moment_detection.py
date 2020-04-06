"""A set of util functions that help to manipulate arrays in any imaginable way."""
import numpy as np

from gaitmap.utils import array_handling


def find_static_samples(
    signal: np.ndarray, window_length: int, overlap: int, inactive_signal_th: float, metric: str = "mean"
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

    overlap : int
        Length of desired overlap in units of samples

    inactive_signal_th: float
       Threshold to decide whether a window should be considered as active or inactive. Window will be tested on
       <= threshold

    metric: str, optional
        Metric which will be calculated per window, one of the following strings

        'mean' (default)
            Calculates mean value per window
        'maximum'
            Calculates maximum value per window
        'median'
            Calculates median value per window
        'variance'
            Calculates variance value per window

    Returns
    -------
    Boolean array with length n to indicate static (=1) or non-static (=0) for each sample

    Examples
    --------
    test_data = load_gyro_data(path)
    get_static_moments(gyro_data, window_length=128, overlap=64, inactive_signal_th = 5, metric = 'mean')

    """
    # TODO: evaluate performance on large datasets

    # test for correct input data shape
    if np.shape(signal)[-1] != 3:
        raise ValueError("Invalid signal dimensions, signal must be of shape (n,3).")

    # supported metric functions
    metric_function = {"maximum": np.max, "variance": np.var, "mean": np.mean, "median": np.median}

    if metric not in metric_function:
        raise ValueError("Invalid metric passed! %s as metric is not supported." % metric)

    # create the list of indices for sliding windows with overlap
    windowed_indices = array_handling.sliding_window_view(
        np.arange(0, len(signal)), window_length, overlap, nan_padding=False
    )

    # allocate output array
    inactive_signal_bool_array = np.zeros(len(signal))

    # calculate norm of input signal (do this outside of loop to boost performance at cost of memory!)
    signal_norm = np.apply_along_axis(np.linalg.norm, 1, signal)

    # iterate over sliding windows
    for indices in windowed_indices:
        # remove potential np.nan entries due to padding
        indices = indices[~np.isnan(indices)].astype(int)

        # fill window with boolean of value comparison
        is_static = metric_function[metric](signal_norm[indices]) <= inactive_signal_th
        bool_window = np.repeat(is_static, len(indices))

        # perform logical or operation to combine all overlapping window results
        inactive_signal_bool_array[indices] = np.logical_or(inactive_signal_bool_array[indices], bool_window)

    return inactive_signal_bool_array


def find_static_sequences(
    signal: np.ndarray, window_length: int, overlap: int, inactive_signal_th: float, metric: str = "mean"
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

    overlap : int
        Length of desired overlap in units of samples

    inactive_signal_th: float
       Threshold to decide whether a window should be considered as active or inactive. Window will be tested on
       <= threshold

    metric: str, optional
        Metric which will be calculated per window, one of the following strings

        'mean' (default)
            Calculates mean value per window
        'maximum'
            Calculates maximum value per window
        'median'
            Calculates median value per window
        'variance'
            Calculates variance value per window

    Returns
    -------
    Array of [start, end] labels indication static regions within the input signal

    Examples
    --------
    gyro_data = load_gyro_data(path)
    static_regions = get_static_moment_labels(gyro_data, window_length=128, overlap=64, inactive_signal_th = 5)

    """
    static_moment_bool_array = find_static_samples(
        signal=signal,
        window_length=window_length,
        overlap=overlap,
        inactive_signal_th=inactive_signal_th,
        metric=metric,
    )
    return array_handling.bool_array_to_start_end_array(static_moment_bool_array)
