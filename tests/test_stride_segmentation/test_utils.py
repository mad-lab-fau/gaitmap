import numpy as np

import pytest

from gaitmap.stride_segmentation.utils import split_array_at_nan, find_local_minima_below_threshold


@pytest.mark.parametrize('data, result', [
    (np.array([np.nan]), []),
    (np.array([1, np.nan, 1]), [[1], [1]]),
    (np.array([1, 2, 3, np.nan, 4, 5]), [[1, 2, 3], [4, 5]]),
    (np.array([np.nan, 1, 2, np.nan, 3, 4, np.nan, 5, 6, np.nan]), [[1, 2], [3, 4], [5, 6]]),
    (np.array([np.nan, 1, 2, np.nan, np.nan, 3, 4, np.nan, np.nan, np.nan, 5, 6, np.nan]), [[1, 2], [3, 4], [5, 6]])
])
def test_split_array_simple(data, result):
    out = split_array_at_nan(data)
    assert len(out) == len(result)

    for a, b in zip(out, result):
        np.testing.assert_array_equal(a[1], np.array(b))


@pytest.mark.parametrize('data, threshold, results',[
    ([*np.ones(10), -1, -2, -1, *np.ones(10)], 0, [11]),
    (2 * [*np.ones(10), -1, -2, -1, *np.ones(10)], 0, [11, 34]),
    ([*np.ones(10), -1, -2, -1, *np.ones(10)], -3, []),
])
def test_find_extrema(data, threshold, results):
    out = find_local_minima_below_threshold(np.array(data), threshold)

    np.testing.assert_array_equal(out, results)
