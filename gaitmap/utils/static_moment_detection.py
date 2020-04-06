"""A set of util functions that help to manipulate arrays in any imaginable way."""
import numpy as np

from gaitmap.utils import array_handling


def get_static_moments(
    signal: np.ndarray, window_length: int, overlap: int, inactive_signal_th: float, metric: str = "mean"
) -> np.ndarray:
    """Search for static windows within given input signal, based on L2-norm thresholding.

    .. warning::
        due to edge cases at the end of the input data where window size and overlap might not fit your data, the last
        window might be discarded for analysis and will therefore always be considered as non-static!

    Parameters
    ----------
    signal : array with shape (n, 3)
        3D signal on which static moment detection should be performed (e.g. 3D-acc or 3D-gyr data)

    window_length : int
        length of desired window

    overlap : int
        length of desired overlap

    inactive_signal_th: float
       threshold to decide whether a window should be considered as active or inactive. Window will be tested on
       <= threshold

    metric: str
        metric which will be calculated per window

    Returns
    -------
    boolean array with length n to indicate static (=1) or non-static (=0) for each sample

    Examples
    --------
    gyro_data = load_gyro_data(path)
    get_static_moments(gyro_data, window_length=128, overlap=64, inactive_signal_th = 5)

    """
    # test for correct input data shape
    if np.shape(signal)[-1] != 3:
        raise ValueError("Invalid sigal dimensions, signal must be of shape (n,3)")

    # create the list of indices for sliding windows with overlap
    windowed_indices = array_handling.sliding_window_view(
        np.arange(0, len(signal)), window_length, overlap, nan_padding=False
    )

    # allocate output array
    inactive_signal_bool_array = np.repeat(0, len(signal))

    # iterate over sliding windows
    for indices in windowed_indices:
        # remove potential np.nan entries due to padding
        indices = indices[~np.isnan(indices)].astype(int)

        # compute norm over multidimensional data (e.g. 3D)
        arr_window_norm = np.apply_along_axis(np.linalg.norm, 1, signal[indices])

        # fill window with boolean of value comparison
        if metric == "mean":
            bool_window = np.repeat(np.mean(arr_window_norm) <= inactive_signal_th, window_length)
        elif metric == "max":
            bool_window = np.repeat(np.max(arr_window_norm) <= inactive_signal_th, window_length)

        # perform logical or operation to combine all overlapping window results
        inactive_signal_bool_array[indices] = np.logical_or(inactive_signal_bool_array[indices], bool_window)
    return inactive_signal_bool_array


def get_static_moment_labels(
    signal: np.ndarray, window_length: int, overlap: int, inactive_signal_th: float, metric: str = "mean"
) -> np.ndarray:
    """Search for static windows within given input signal, based on L2-norm thresholding.

    .. warning::
        due to edge cases at the end of the input data where window size and overlap might not fit your data, the last
        window might be discarded for analysis and will therefore always be considered as non-static!

    Parameters
    ----------
    signal : array with shape (n, 3)
        3D signal on which static moment detection should be performed (e.g. 3D-acc or 3D-gyr data)

    window_length : int
        length of desired window

    overlap : int
        length of desired overlap

    inactive_signal_th: float
       threshold to decide whether a window should be considered as active or inactive. Window will be tested on
       <= threshold

    metric: str
        metric which will be calculated per window

    Returns
    -------
    list of [start, stop] labels indication static regions within the input signal

    Examples
    --------
    gyro_data = load_gyro_data(path)
    static_regions = get_static_moment_labels(gyro_data, window_length=128, overlap=64, inactive_signal_th = 5)

    """
    static_moment_bool_array = get_static_moments(
        signal=signal,
        window_length=window_length,
        overlap=overlap,
        inactive_signal_th=inactive_signal_th,
        metric=metric,
    )
    return array_handling.bool_array_to_start_stop_list(static_moment_bool_array)
