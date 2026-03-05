"""Vendored functions from tslearn.

This is a set of functions directly copied from the DTW implementation of tslearn.
The original implementations can be found here:
https://github.com/tslearn-team/tslearn/blob/9d51f9a71c632ce9b38e0b08cb41583e2b0de4f4/tslearn/metrics/dtw_variants.py

tslearn (and hence this file) is licensed under BSD 2-Clause.
All copyright remains with the original authors (Copyright (c) 2017, Romain Tavenard).
For more information see the LICENSE file in the tslearn repository.
https://github.com/tslearn-team/tslearn/blob/5504b02ed4ce91ae8393959d29a9f5259a6ba977/LICENSE
"""

import numpy
from numba import njit


@njit()
def _local_squared_dist(x, y):
    dist = 0.0
    for di in range(x.shape[0]):
        diff = x[di] - y[di]
        dist += diff * diff
    return dist


@njit()
def _subsequence_cost_matrix(subseq, longseq):
    l1 = subseq.shape[0]
    l2 = longseq.shape[0]
    cum_sum = numpy.full((l1 + 1, l2 + 1), numpy.inf)
    cum_sum[0, :] = 0.0

    for i in range(l1):
        for j in range(l2):
            cum_sum[i + 1, j + 1] = _local_squared_dist(subseq[i], longseq[j])
            cum_sum[i + 1, j + 1] += min(cum_sum[i, j + 1], cum_sum[i + 1, j], cum_sum[i, j])
    return cum_sum[1:, 1:]


def subsequence_cost_matrix(subseq, longseq):
    """Compute the accumulated cost matrix score between a subsequence and a reference time series.

    Parameters
    ----------
    subseq : array, shape = (sz1, d)
        Subsequence time series.

    longseq : array, shape = (sz2, d)
        Reference time series

    Returns
    -------
    mat : array, shape = (sz1, sz2)
        Accumulated cost matrix.

    """
    return _subsequence_cost_matrix(subseq, longseq)


@njit()
def _subsequence_path(acc_cost_mat, idx_path_end):
    sz1, _ = acc_cost_mat.shape
    path = [(sz1 - 1, idx_path_end)]
    while path[-1][0] != 0:
        i, j = path[-1]
        if i == 0:
            path.append((0, j - 1))
        elif j == 0:
            path.append((i - 1, 0))
        else:
            arr = numpy.array([acc_cost_mat[i - 1][j - 1], acc_cost_mat[i - 1][j], acc_cost_mat[i][j - 1]])
            argmin = numpy.argmin(arr)
            if argmin == 0:
                path.append((i - 1, j - 1))
            elif argmin == 1:
                path.append((i - 1, j))
            else:
                path.append((i, j - 1))
    return path[::-1]


def subsequence_path(acc_cost_mat, idx_path_end):
    r"""Compute the optimal path through a accumulated cost matrix given the endpoint of the sequence.

    Parameters
    ----------
    acc_cost_mat: array, shape = (sz1, sz2)
        The accumulated cost matrix comparing subsequence from a longer
        sequence.
    idx_path_end: int
        The end position of the matched subsequence in the longer sequence.

    Returns
    -------
    path: list of tuples of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to `subseq` and the second one corresponds to
        `longseq`. The startpoint of the Path is :math:`P_0 = (0, ?)` and it
        ends at :math:`P_L = (len(subseq)-1, idx\_path\_end)`

    Examples
    --------
    >>> acc_cost_mat = numpy.array([[1.0, 0.0, 0.0, 1.0, 4.0], [5.0, 1.0, 1.0, 0.0, 1.0]])
    >>> # calculate the globally optimal path
    >>> optimal_end_point = numpy.argmin(acc_cost_mat[-1, :])
    >>> path = subsequence_path(acc_cost_mat, optimal_end_point)
    >>> path
    [(0, 2), (1, 3)]

    """
    return _subsequence_path(acc_cost_mat, idx_path_end)
