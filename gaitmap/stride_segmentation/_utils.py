"""Some general utils for the stride segmentation algorithms."""

from typing import Tuple, Union

import numpy as np

from gaitmap.utils.array_handling import find_extrema_in_radius


def snap_to_min(
    data: np.ndarray,
    matches_start_end: np.ndarray,
    snap_to_min_win_samples: Union[int, Tuple[int, int]],
):
    """Post process a set of matches by "snapping" their start and end values to the closest minima of the data.

    Parameters
    ----------
    data
        The raw IMU data that should be used for snapping. Should be a 1D numpy array
    matches_start_end
        The actual stride candidates as 2D np.array
    snap_to_min_win_samples
        If a single integer is provided, this is the size of the window that should be searched for the closest
        extrema in samples.
        The window is assumed to be centered around the start/end value of the match (i.e. the search radius is half
        the window size)
        If a tuple of two integers is provided, the window is assumed to be asymmetric and the first value is the
        search radius to the left of the start/end value and the second value is the search radius to the right of
        the start/end value.
        Note: In case you need different search intervals for the start and end values of a stride, you can call this
        function twice, only with the start or end values of the stride candidates and change the search radius
        accordingly.

    """
    if isinstance(snap_to_min_win_samples, int):
        snap_to_min_win_samples = (snap_to_min_win_samples // 2, snap_to_min_win_samples // 2)

    # Find the closest minimum for each start and stop value
    flattened_matches = matches_start_end.flatten()
    # Because the actual end values are exclusive, we need to handle the case were the stride ends inclusive the
    # last value
    edge_case_stride = flattened_matches == len(data)
    flattened_matches[edge_case_stride] -= 1
    flattened_matches = find_extrema_in_radius(
        data,
        flattened_matches,
        snap_to_min_win_samples,
    )
    # All strides that were inclusive with the last sample and didn't change the sample will be changed back
    # to be inclusive.
    # Strides that were "snapped" to the last sample are exclusive the last sample.
    # Their remains an edge case were a stride that was inclusive the last sample was correctly snapped to be
    # exclusive and is then updated to be inclusive again in the following line.
    # However, this is not worth handling.
    flattened_matches[edge_case_stride & (flattened_matches == len(data) - 1)] += 1
    matches_start_end = flattened_matches.reshape(matches_start_end.shape)
    return matches_start_end
