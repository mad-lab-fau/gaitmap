"""A set of util functions that help to find static windows in imu signals."""
import numpy as np


def sliding_window_view(data: np.ndarray, window_length: int, window_overlap: int, copy: bool = False) -> np.ndarray:
    """Create a sliding window view of an input array with given window length and overlap.

    Parameters
    ----------
    data : np.ndarray
        input data on which sliding window action should be performed, array with shape (1,) or (n, 3)

    window_length : int
        length of a desired window in samples

    window_overlap : int
        length of desired overlap in samples

    copy : bool
        select if you want to get just the view or a copy of the data

    Returns
    -------
    np.ndarray : windowed view or windowed copy of input array

    Examples
    --------
    Sliding window of 4 samples and 50% overlap
    >>> data = np.arange(0,10)
    >>> sliding_window_view(data = data, window_length = 4, window_overlap = 2)

    """
    if window_overlap >= window_length:
        raise ValueError("Invalid Input, overlap must be smaller than window length!")

    if window_length < 2:
        raise ValueError("Invalid Input, window_length must be larger than 1!")

    # check if we need to pad the data at its end to match window and overlap
    if len(data) % (window_length - window_overlap):
        data = np.append(data, np.repeat(np.nan, window_length - window_overlap))

    if window_overlap == 0:
        window_overlap = window_length

    sh = (data.size - window_length + 1, window_length)
    st = data.strides * 2
    view = np.lib.stride_tricks.as_strided(data, strides=st, shape=sh)[0::window_overlap]
    if copy:
        return view.copy()

    return view
