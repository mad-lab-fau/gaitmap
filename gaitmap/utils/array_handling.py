"""A set of util functions that help to manipulate arrays in any imaginable way."""

from collections.abc import Iterable, Iterator
from typing import Optional, Union

import numba.typed
import numpy as np
from numba import njit
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from typing_extensions import Literal

from gaitmap.utils.datatype_helper import (
    SingleSensorData,
    SingleSensorRegionsOfInterestList,
    SingleSensorStrideList,
    is_single_sensor_data,
    is_single_sensor_regions_of_interest_list,
    is_single_sensor_stride_list,
)
from gaitmap.utils.exceptions import ValidationError


def sliding_window_view(arr: np.ndarray, window_length: int, overlap: int, nan_padding: bool = False) -> np.ndarray:
    """Create a sliding window view of an input array with given window length and overlap.

    .. warning::
       This function will return by default a view onto your input array, modifying values in your result will directly
       affect your input data which might lead to unexpected behaviour! If padding is disabled (default) last window
       fraction of input may not be returned! However, if `nan_padding` is enabled, this will always return a copy
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
    >>> data = np.arange(0, 10)
    >>> windowed_view = sliding_window_view(arr=data, window_length=5, overlap=3, nan_padding=True)
    >>> windowed_view
    array([[ 0.,  1.,  2.,  3.,  4.],
           [ 2.,  3.,  4.,  5.,  6.],
           [ 4.,  5.,  6.,  7.,  8.],
           [ 6.,  7.,  8.,  9., nan]])

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

    The end index is the first element after the True-region,
    so you can use it for upper-bound exclusive slicing etc.

    Parameters
    ----------
    bool_array : array with shape (n,)
        boolean array with either 0/1, 0.0/1.0 or True/False elements

    Returns
    -------
    array of [start, end] indices with shape (n,2)

    Examples
    --------
    >>> example_array = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1])
    >>> start_end_list = bool_array_to_start_end_array(example_array)
    >>> start_end_list
    array([[2, 4],
           [6, 9]])
    >>> example_array[start_end_list[0, 0] : start_end_list[0, 1]]
    array([1, 1])

    """
    # check if input is a numpy array. E.g. for a Series, start and end will just be the first and last index.
    if not isinstance(bool_array, np.ndarray):
        raise TypeError("Input must be a numpy array!")

    # check if input is actually a boolean array
    if not np.array_equal(bool_array, bool_array.astype(bool)):
        raise ValueError("Input must be boolean array!")

    if len(bool_array) == 0:
        return np.array([])

    slices = np.ma.flatnotmasked_contiguous(np.ma.masked_equal(bool_array, 0))
    return np.array([[s.start, s.stop] for s in slices])


def start_end_array_to_bool_array(start_end_array: np.ndarray, pad_to_length: Optional[int] = None) -> np.ndarray:
    """Convert a start-end list to a bool array.

    Parameters
    ----------
    start_end_array : array with shape (n,2)
        2d-array indicating start and end values e.g. [[10,20],[20,40],[70,80]]

        .. warning:: We assume that the end index is exclusiv!
                     This is in line with the definitions of stride and roi lists in gaitmap.

    pad_to_length: int
        Define the length of the resulting array.
        If None, the array will have the length of the largest index.
        Otherwise, the final array will either be padded with False or truncated to the specified length.

    Returns
    -------
    array with shape (n,)
        boolean array with True/False elements

    Examples
    --------
    >>> import numpy as np
    >>> example_array = np.array([[3, 5], [7, 8]])
    >>> start_end_array_to_bool_array(example_array, pad_to_length=12)
    array([False, False, False,  True,  True,  True, False,  True,  True,
           False, False, False])
    >>> example_array = np.array([[3, 5], [7, 8]])
    >>> start_end_array_to_bool_array(example_array, pad_to_length=None)
    array([False, False, False,  True,  True,  True, False,  True,  True])

    """
    start_end_array = np.atleast_2d(start_end_array)

    if pad_to_length is None:
        n_elements = start_end_array.max()
    else:
        if pad_to_length < 0:
            raise ValueError("pad_to_length must be positive!")
        n_elements = pad_to_length

    bool_array = np.zeros(n_elements)
    for start, end in start_end_array:
        bool_array[start:end] = 1
    return bool_array.astype(bool)


def split_array_at_nan(a: np.ndarray) -> list[tuple[int, np.ndarray]]:
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
        # If not None, take the negative value.
        # If None, just pass it like it is to find_peaks
        threshold *= -1
    return find_peaks(-data, height=threshold, **kwargs)[0]


def find_extrema_in_radius(
    data: np.ndarray,
    indices: np.ndarray,
    radius: Union[int, tuple[int, int]],
    extrema_type: Literal["min", "max"] = "min",
):
    """Return the index of the global extrema of data in the given radius around each index in indices.

    Parameters
    ----------
    data : 1D array
        Data used to find the extrema
    indices : 1D array of ints
        Around each index the extremum is searched in the region defined by radius
    radius
        The number of samples to the left and the right that are considered for the search.
        If a single integer is given, the same radius is used for both sides.
        The final search window has the length 2 * radius + 1.
        If a tuple of two integers is given, the first integer is used for the left side and the second for the right
        side.
        The final search window has the length left_radius + right_radius + 1.
        In case the radius is 0 (or a tuple of 0, 0), the indices are returned without further processing.
    extrema_type
        If the minima or maxima of the data are searched.

    Returns
    -------
    list_of_extrema_indices
        Array of the position of each identified extremum

    """
    extrema_funcs = {"min": np.nanargmin, "max": np.nanargmax}
    if extrema_type not in extrema_funcs:
        raise ValueError(f"`extrema_type` must be one of {list(extrema_funcs.keys())}, not {extrema_type}")
    extrema_func = extrema_funcs[extrema_type]
    if radius in (0, (0, 0)):
        # In case the search radius is 0 samples, we can just return the input.
        return indices
    if isinstance(radius, int):
        before, after = radius, radius
    elif isinstance(radius, tuple):
        before, after = radius
    else:
        raise TypeError(f"radius must be int or a tuple of two ints, not {radius}")

    # Search region is twice the sum of indices before and after + 1
    d = before + after + 1
    start_padding = 0

    data = data.astype(float)
    # If one of the indices is to close to the end of the data, we need to pad the data
    if len(data) - np.max(indices) <= after:
        data = np.pad(data, (0, after), constant_values=np.nan)
    # If one of the indices is to close to the beginning of the data, we need to pad the data
    if np.min(indices) < before:
        start_padding = before
        data = np.pad(data, (start_padding, 0), constant_values=np.nan)
    strides = sliding_window_view(data, window_length=d, overlap=d - 1)
    # select all windows around indices
    actual_window_start = indices.astype("int64") - before + start_padding
    windows = strides[actual_window_start, :]
    return extrema_func(windows, axis=1) + actual_window_start - start_padding


@njit(cache=True)
def _bool_fill(indices: np.ndarray, bool_values: np.ndarray, array: np.ndarray) -> np.ndarray:
    """Fill a preallocated array with bool_values.

    This method iterates over the indices and adds the values to the array at the given indices using a logical or.
    """
    for i, idx in enumerate(indices):
        index = idx
        val = bool_values[i]
        index = index[~np.isnan(index)]
        # perform logical or operation to combine all overlapping window results
        array[index] = np.logical_or(array[index], val)
    return array


def multi_array_interpolation(arrays: list[np.ndarray], n_samples, kind: str = "linear") -> np.ndarray:
    """Interpolate multiple 2D-arrays to the same length along axis 0.

    Parameters
    ----------
    arrays
        List of 2D arrays.
        Note that `arr.shape[1]` must be identical for all arrays.
    n_samples
        Number of samples the arrays should be resampled to.
    kind
        The type of interpolation to use.
        Refer to :func:`~scipy.interpolate.interp1d` for possible options.

        .. note:: In case of linear interpolation a custom numba accelerated method is used that is significantly
                  faster than the scipy implementation.

    Returns
    -------
    interpolated_data
        result as a single 3-D array of shape `(len(arrays), arrays[0].shape[1], n_samples)`.

    """
    if kind == "linear":
        arrays = numba.typed.List(arrays)
        return _fast_linear_interpolation(arrays, n_samples)
    final_array = np.empty((len(arrays), arrays[0].shape[1], n_samples))
    for i, s in enumerate(arrays):
        x_orig = np.linspace(0, len(s), len(s))
        x_new = np.linspace(0, len(s), n_samples)
        final_array[i] = interp1d(x_orig, s.T, kind=kind)(x_new)
    return final_array


@numba.njit()
def _fast_linear_interpolation(arrays: numba.typed.List, n_samples) -> np.ndarray:
    final_array = np.empty((len(arrays), arrays[0].shape[1], n_samples))
    for i, s in enumerate(arrays):
        s_len = len(s)
        x_orig = np.linspace(0, s_len, s_len)
        x_new = np.linspace(0, s_len, n_samples)
        for j, col in enumerate(s.T):
            final_array[i, j] = np.interp(x_new, x_orig, col)
    return final_array


def merge_intervals(input_array: np.ndarray, gap_size: int = 0) -> np.ndarray:
    """Merge intervals that are overlapping and that are a distance less or equal to gap_size from each other.

    This is actually a wrapper for _solve_overlap that is needed because numba can not compile np.sort().

    Parameters
    ----------
    input_array : (n, 2) np.ndarray
        The np.ndarray containing the intervals that should be merged
    gap_size : int
        Integer that sets the allowed gap between intervals.
        For examples see below.
        Default is 0.

    Returns
    -------
    merged intervals array
        (n, 2) np.ndarray containing the merged intervals

    Examples
    --------
    >>> test = np.array([[1, 3], [2, 4], [6, 8], [5, 7], [10, 12], [11, 15], [18, 20]])
    >>> merge_intervals(test)
    array([[ 1,  4],
           [ 5,  8],
           [10, 15],
           [18, 20]])

    >>> merge_intervals(test, 2)
    array([[ 1, 15],
           [18, 20]])

    """
    if input_array.shape[0] == 0:
        return input_array

    return np.array(_solve_overlap(np.sort(input_array, axis=0, kind="stable"), gap_size))


@njit
def _solve_overlap(input_array: np.ndarray, gap_size: int) -> numba.typed.List:
    """Merge intervals that are overlapping and that are a distance less or equal to gap_size from each other."""
    stack = numba.typed.List()
    stack.append(input_array[0])

    for i in range(1, len(input_array)):
        if stack[-1][0] <= input_array[i][0] <= (stack[-1][1] + gap_size) <= (input_array[i][1] + gap_size):
            stack[-1][1] = input_array[i][1]
        else:
            stack.append(input_array[i])

    return stack


def iterate_region_data(
    signal_sequence: Iterable[SingleSensorData],
    label_sequences: Iterable[Union[SingleSensorStrideList, SingleSensorRegionsOfInterestList]],
    expected_col_order: Optional[list[str]] = None,
) -> Iterator[SingleSensorData]:
    """Iterate over individual strides/ROIs in multiple sensor data sequences.

    This can be helpful if you want to apply operations to each of these strides or ROIs and there is no other way
    besides iterating over the data.

    Parameters
    ----------
    signal_sequence : Iterable[SingleSensorData]
        A list/iterable of SingleSensorData objects.
    label_sequences : Iterable[Union[SingleSensorStrideList, SingleSensorRegionsOfInterestList]]
        A list/iterable of SingleSensorStrideList or SingleSensorRegionsOfInterestList objects.
        This must have the same length as `signal_sequence` as we expect that the first signal sequence belongs to the
        first label sequence.
    expected_col_order : Optional[List[str]]
        A list of strings that defines the expected column order of the output.
        This ensures that all signal sequences have the same column order.
        If None, the column order of the first signal sequence is used.

    Yields
    ------
    SingleSensorData
        A SingleSensorData object with the data of the next stride/ROI.

    """
    for df, labels in zip(signal_sequence, label_sequences):
        is_single_sensor_data(df, check_acc=False, check_gyr=False, raise_exception=True)
        try:
            is_single_sensor_stride_list(labels, raise_exception=True)
        except ValidationError as e_stride_list:
            try:
                is_single_sensor_regions_of_interest_list(labels, raise_exception=True)
            except ValidationError as e_roi:
                raise ValidationError(
                    "The label sequences must be either SingleSensorStrideList or SingleSensorRegionsOfInterestList. "
                    "\n"
                    "The validations failed with the following errors: \n\n"
                    f"Stride List: \n\n{e_stride_list!s}\n\n"
                    f"Regions of Interest List: \n\n{e_roi!s}"
                ) from e_roi
        if expected_col_order is None:
            # In the first iteration we pull the column order.
            # We don't do that beforehand, because otherwise we could not take a generator as input and force the
            # user to put all the data in RAM first.
            expected_col_order = df.columns
        df = df.reindex(columns=expected_col_order)
        for _, s, e in labels[["start", "end"]].itertuples():
            yield df.iloc[s:e]
