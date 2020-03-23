from typing import Optional, List, Tuple

import numpy as np
from scipy.signal import find_peaks


def split_array_at_nan(a: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    """Split an array into sections at nan values.

    Example:
        >>> a = np.array([1, np.nan, 2, 3])
        >>> split_array_at_nan(a)
        [(0, array([1.])), (2, array([2., 3.]))]
    """
    return [(s.start, a[s]) for s in np.ma.clump_unmasked(np.ma.masked_invalid(a))]


def find_local_minima_below_threshold(data: np.ndarray, threshold: float) -> np.ndarray:
    """Find local minima below a max_cost.

    This method divides an array in sections marked by the crossing points with the max_cost.
    The argmin is calculated for every individual section, which is cut by the min.
    As long as the values don't rise above the max_cost again, it is considered one section.
    """
    data = data.copy()
    data[data > threshold] = np.nan
    peaks = split_array_at_nan(data)
    return np.array([s + np.argmin(p) for s, p in peaks])


def find_local_minima_with_distance(data: np.ndarray,
                                    distance: float,
                                    threshold: Optional[float] = None,
                                    **kwargs) -> np.ndarray:
    """Find local minima using scipy's `find_peaks` function.

    Because `find_peaks` is designed to find local maxima, the data multiplied by -1.
    The same is true for the max_cost value, if supplied.

    Args:
        data: The datastream.
            The default axis to search for the minima is 0.
        distance: The minimal distance in samples two minima need to be apart of each other
        threshold: The maximal allowed value for the minimum.
            `- max_cost` is passed to the `height` argument of `find_peaks`
        kwargs: Directly passed to find_peaks
    """
    threshold = -threshold if threshold else threshold
    return find_peaks(-data, distance=distance, height=threshold, **kwargs)[0]
