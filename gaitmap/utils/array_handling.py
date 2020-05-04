"""A set of util functions that help to manipulate arrays in any imaginable way."""
from typing import List, Tuple, Optional

import numpy as np
from numba import njit
from scipy.signal import find_peaks


def sliding_window_view(arr: np.ndarray, window_length: int, overlap: int, nan_padding: bool = False) -> np.ndarray:
    """Create a sliding window view of an input array with given window length and overlap.

    .. warning::
        This function will return by default a view onto your input array, modifying values in your result will directly
        affect your input data which might lead to unexpected behaviour! If padding is disabled (default) last window
        fraction of input may not be returned! However, if nan_padding is enabled, this will always return a copy
        instead of a view of your input data, independent if padding was actually performed or not!

    Parameters
    ----------
    arr : array with shape (n,) or (n, m)
        array on which sliding window action should be performed. Windowing
        will always be performed along axis 0.

    window_length : int
        length of desired window (must be smaller than array length n)

    overlap : int
        length of desired overlap (must be smaller than window_length)

    nan_padding: bool
        select if last window should be nan-padded or discarded if it not fits with input array length. If nan-padding
        is enabled the return array will always be a copy of the input array independent if padding was actually
        performed or not!

    Returns
    -------
    windowed view (or copy for nan_padding) of input array as specified, last window might be nan padded if necessary to
    match window size

    Examples
    --------
    >>> data = np.arange(0,10)
    >>> windowed_view = sliding_window_view(arr = data, window_length = 5, overlap = 3, nan_padding = True)
    >>> windowed_view
    np.array([[0, 1, 2, 3, 4], [2, 3, 4, 5, 6], [4, 5, 6, 7, 8], [6, 7, 8, 9, np.nan]])

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
        if nan_padding:
            # np.pad always returns a copy of the input array even if pad_length is 0!
            arr = np.pad(arr.astype(float), (0, pad_length), constant_values=np.nan)

        new_shape = (arr.size - window_length + 1, window_length)
    else:
        if nan_padding:
            # np.pad always returns a copy of the input array even if pad_length is 0!
            arr = np.pad(arr.astype(float), [(0, pad_length), (0, 0)], constant_values=np.nan)

        shape = (window_length, arr.shape[-1])
        n = np.array(arr.shape)
        o = n - shape + 1  # output shape
        new_shape = np.concatenate((o, shape), axis=0)

    # apply stride_tricks magic
    new_strides = np.concatenate((arr.strides, arr.strides), axis=0)
    view = np.lib.stride_tricks.as_strided(arr, new_shape, new_strides)[0 :: (window_length - overlap)]

    view = np.squeeze(view)  # get rid of single-dimensional entries from the shape of an array.

    return view


def bool_array_to_start_end_array(bool_array: np.ndarray) -> np.ndarray:
    """Find regions in bool array and convert those to start-end indices.

    Parameters
    ----------
    bool_array : array with shape (n,)
        boolean array with either 0/1, 0.0/1.0 or True/False elements

    Returns
    -------
    array of [start, end] indices with shape (n,2)

    Examples
    --------
    >>> example_array = np.array([0,0,1,1,0,0,1,1,1])
    >>> start_end_list = bool_array_to_start_end_array(example_array)
    array([[2, 3],[6, 8]])

    """
    # check if input is actually a boolean array
    if not np.array_equal(bool_array, bool_array.astype(bool)):
        raise ValueError("Input must be boolean array!")

    slices = np.ma.flatnotmasked_contiguous(np.ma.masked_equal(bool_array, 0))
    return np.array([[s.start, s.stop - 1] for s in slices])


def split_array_at_nan(a: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    """Split an array into sections at nan values.

    Examples
    --------
    >>> a = np.array([1, np.nan, 2, 3])
    >>> split_array_at_nan(a)
    [(0, array([1.])), (2, array([2., 3.]))]

    """
    return [(s.start, a[s]) for s in np.ma.clump_unmasked(np.ma.masked_invalid(a))]


def find_local_minima_below_threshold(data: np.ndarray, threshold: float) -> np.ndarray:
    """Find local minima below a max_cost.

    This method divides an array in sections marked by the crossing points with the threshold.
    The argmin is calculated for every individual section.
    As long as the values don't rise above the threshold again, it is considered one section.
    """
    data = data.copy()
    data[data > threshold] = np.nan
    peaks = split_array_at_nan(data)
    return np.array([s + np.argmin(p) for s, p in peaks])


def find_local_minima_with_distance(data: np.ndarray, threshold: Optional[float] = None, **kwargs) -> np.ndarray:
    """Find local minima using scipy's `find_peaks` function.

    Because `find_peaks` is designed to find local maxima, the data multiplied by -1.
    The same is true for the threshold value, if supplied.

    Parameters
    ----------
    data
        The datastream.
        The default axis to search for the minima is 0.
        To search for minima this is multiplied by -1 before passing to `find_peaks`
    threshold
        The maximal allowed value for the minimum.
        `- threshold` is passed to the `height` argument of `find_peaks`
    kwargs
        Directly passed to find_peaks

    """
    if threshold:
        # If not None take the negative value.
        # If None just pass it like it is to find_peaks
        threshold *= -1
    return find_peaks(-data, height=threshold, **kwargs)[0]


def find_minima_in_radius(data: np.ndarray, indices: np.ndarray, radius: int):
    """Return the index of the global minima of data in the given radius around each index in indices.

    Parameters
    ----------
    data : 1D array
        Data used to find the minima
    indices : 1D array of ints
        Around each index the minima is searched in the region defined by radius
    radius
        The number of samples to the left and the right that are considered for the search.
        The final search window has the length 2 * radius + 1
    Returns
    -------
    list_of_minima_indices
        Array of the position of each identified minima

    """
    # Search region is twice the radius centered around each index
    data = data.astype(float)
    d = 2 * radius + 1
    start_padding = 0
    if len(data) - np.max(indices) <= radius:
        data = np.pad(data, (0, radius), constant_values=np.nan)
    if np.min(indices) < radius:
        start_padding = radius
        data = np.pad(data, (start_padding, 0), constant_values=np.nan)
    strides = sliding_window_view(data, window_length=d, overlap=d - 1)
    # select all windows around indices
    windows = strides[indices.astype(int) - radius + start_padding, :]
    return np.nanargmin(windows, axis=1) + indices - radius


@njit(cache=True)
def _bool_fill(indices: np.ndarray, bool_values: np.ndarray, array: np.ndarray) -> np.ndarray:
    """Fill a preallocated array with bool_values.

    This method iterates over the indices and adds the values to the array at the given indices using a logical or.
    """
    for i in range(len(indices)):  # noqa: consider-using-enumerate
        index = indices[i]
        val = bool_values[i]
        index = index[~np.isnan(index)]
        # perform logical or operation to combine all overlapping window results
        array[index] = np.logical_or(array[index], val)
    return array
