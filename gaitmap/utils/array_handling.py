"""A set of util functions that help to manipulate arrays in any imaginable way."""
import numpy as np


def sliding_window_view(arr: np.ndarray, window_length: int, overlap: int) -> np.ndarray:
    """Create a sliding window view of an input array with given window length and overlap.

    Parameters
    ----------
    arr : array with shape (n,) or (n, m)
        array on which sliding window action should be performed. Windowing
        will always be performed along axis 0.

    window_length : int
        length of desired window

    overlap : int
        length of desired overlap

    Returns
    -------
    windowed view of input array as specified

    Examples
    --------
    >>> #Sliding window of 4 samples and 50% overlap 1D array
    >>> data = np.arange(0,10)
    >>> windowed_view = sliding_window_view(arr = data, window_length = 4, overlap = 2)
    >>> #Sliding window of 7 samples and 2 sample overlap 3D array
    >>> data = np.column_stack([np.arange(0, 10), np.arange(0, 10), np.arange(0, 10)])
    >>> windowed_view = sliding_window_view(arr = data, window_length = 7, overlap = 2)

    """
    if overlap >= window_length:
        raise ValueError("Invalid Input, overlap must be smaller than window length!")

    if window_length < 2:
        raise ValueError("Invalid Input, window_length must be larger than 1!")

    # calculate length of necessary np.nan-padding to make sure windows and overlaps exactly fits data length
    n_windows = np.ceil((len(arr) - window_length) / (window_length - overlap)).astype(int)
    pad_length = window_length + n_windows * (window_length - overlap) - len(arr)

    # had to handle 1D arrays separately
    if arr.ndim == 1:
        pad = np.repeat(np.nan, pad_length)
        arr = np.append(arr, pad)

        new_shape = (arr.size - window_length + 1, window_length)
    else:
        pad = np.ones((pad_length, np.shape(arr)[-1])) * np.nan
        arr = np.append(arr, pad, axis=0)

        shape = (window_length, arr.shape[-1])
        n = np.array(arr.shape)
        o = n - shape + 1  # output shape
        new_shape = np.concatenate((o, shape), axis=0)

    # apply stride_tricks magic
    new_strides = np.concatenate((arr.strides, arr.strides), axis=0)
    view = np.lib.stride_tricks.as_strided(arr, new_shape, new_strides)[0 :: (window_length - overlap)]

    return np.squeeze(view)  # get rid of single-dimensional entries from the shape of an array.
