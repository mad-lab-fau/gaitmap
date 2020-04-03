"""A set of util functions that help to find static windows in imu signals."""
import numpy as np


def sliding_window_view(data: np.ndarray, window_length: int, window_overlap: int) -> np.ndarray:
    """Create a sliding window view of an input array with given window length and overlap.

    Parameters
    ----------
    data : np.ndarray
        input data on which sliding window action should be performed, array with shape (1,) or (n, m) while windowing
        will be performed along axis 0

    window_length : int
        length of a desired window in samples

    window_overlap : int
        length of desired overlap in samples

    Returns
    -------
    np.ndarray : windowed view or windowed copy of input array

    Examples
    --------
    Sliding window of 4 samples and 50% overlap 1D array
    >>> data = np.arange(0,10)
    >>> windowed_view = sliding_window_view(data = data, window_length = 4, window_overlap = 2)

    Sliding window of 7 samples and 2 sample overlap 3D array
    >>> data = np.column_stack([np.arange(0, 10), np.arange(0, 10), np.arange(0, 10)])
    >>> windowed_view = sliding_window_view(data = data, window_length = 7, window_overlap = 2)

    """
    if window_overlap >= window_length:
        raise ValueError("Invalid Input, overlap must be smaller than window length!")

    if window_length < 2:
        raise ValueError("Invalid Input, window_length must be larger than 1!")

    # calculate length of necessary np.nan-padding to make sure windows and overlaps exactly fits data length
    n_windows = np.ceil((len(data) - window_length) / (window_length - window_overlap)).astype(int)
    pad_length = window_length + n_windows * (window_length - window_overlap) - len(data)

    # had to handle 1D arrays separately
    if data.ndim == 1:
        pad = np.repeat(np.nan, pad_length)
        data = np.append(data, pad)

        new_shape = (data.size - window_length + 1, window_length)
    else:
        pad = np.ones((pad_length, np.shape(data)[-1])) * np.nan
        data = np.append(data, pad, axis=0)

        shape = (window_length, data.shape[-1])
        n = np.array(data.shape)
        o = n - shape + 1  # output shape
        new_shape = np.concatenate((o, shape), axis=0)

    # apply stride_tricks magic
    new_strides = np.concatenate((data.strides, data.strides), axis=0)
    view = np.lib.stride_tricks.as_strided(data, new_shape, new_strides)[0:: (window_length - window_overlap)]

    return np.squeeze(view)  # get rid of single-dimensional entries from the shape of an array.
