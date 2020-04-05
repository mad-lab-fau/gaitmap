"""A set of util functions that help to manipulate arrays in any imaginable way."""
import numpy as np

from gaitmap.utils import array_handling


def static_moment_detection(
    signal: np.ndarray, window_length: int, overlap: int, inactive_signal_th: float
) -> np.ndarray:
    """Create a sliding window view of an input array with given window length and overlap.

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

    Returns
    -------
    boolean array with length n to indicate activity or inactivity for each sample

    Examples
    --------
    TBD

    """
    # create the list of indices for sliding windows with overlap
    windowed_indices = array_handling.sliding_window_view(np.arange(0, len(signal)), window_length, overlap, nan_padding=True)

    # allocate output array
    inactive_signal_bool_array = np.zeros(len(signal))

    # iterate over sliding windows
    for window_indices in windowed_indices:
        # remove potential np.nan entries due to padding
        window_indices = window_indices[~np.isnan(window_indices)]
        # compute signal energy based on the whole gyro signal (3d)
        arr_window = signal[window_indices]

        # compute norm over multidimensional data (e.g. 3D)
        arr_window_norm = np.apply_along_axis(np.linalg.norm, 1, arr_window)

        # if mean of windowed input signal norm is smaller than given threshold, consider this window as inactive
        if np.nanmean(arr_window_norm) >= inactive_signal_th:

            inactive_signal_bool_array[window_indices] = 1

    return inactive_signal_bool_array
