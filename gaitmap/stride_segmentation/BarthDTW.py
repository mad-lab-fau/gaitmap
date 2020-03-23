from typing import Optional, Sequence, List

import numpy as np
from scipy.signal import resample, find_peaks
from tslearn.metrics import subsequence_cost_matrix, subsequence_path
from tslearn.utils import to_time_series

from gaitmap.base import BaseStrideSegmentation


def find_local_minima_with_distance(
    data: np.ndarray, distance: float, max_cost: Optional[float] = None, **kwargs
) -> np.ndarray:
    """Find local minima using scipy's `find_peaks` function.

    Because `find_peaks` is designed to find local maxima, the data multiplied by -1.
    The same is true for the height value, if supplied.

    Args:
        data
            The datastream. The default axis to search for the minima is 0.
        distance
            The minimal distance in samples two minima need to be apart of each other
        max_cost:
            The maximal allowed value for the minimum. `- max_cost` is passed to the `height` argument of `find_peaks`
        kwargs:
            Directly passed to find_peaks
    """
    max_cost = -max_cost if max_cost else max_cost
    return find_peaks(-data, distance=distance, height=max_cost, **kwargs)[0]


class BarthDTW(BaseStrideSegmentation):
    """Segment strides using a single stride template and Dynamic Time Warping."""
    template: np.ndarray
    template_sampling_rate: float
    interpolate_template: bool
    threshold: float
    min_stride_time_s: float

    acc_cost_mat_: Optional[np.ndarray] = None
    paths_: Optional[Sequence[Sequence[tuple]]] = None
    costs_: Optional[Sequence[float]] = None

    data: np.ndarray
    sampling_rate: float

    @property
    def path_start_stops_(self) -> List[List[int]]:
        return [[p[0][-1], p[-1][-1]] for p in self.paths_]

    @property
    def cost_function_(self):
        return np.sqrt(self.acc_cost_mat_[-1, :])

    def __init__(
        self,
        template: np.ndarray,
        template_sampling_rate: float,
        threshold: float,
        min_stride_time_s: float = 0.6,
        interpolate_template: bool = True,
    ):
        self.template = template
        self.template_sampling_rate = template_sampling_rate
        self.interpolate_template = interpolate_template
        self.threshold = threshold
        self.min_stride_time = min_stride_time_s

    def segment(self, data: np.ndarray, sampling_rate: float, **kwargs) -> List[List[int]]:
        self.data = data
        self.sampling_rate = sampling_rate
        template = self._interpolate_template(sampling_rate)

        self.acc_cost_mat_ = self._calculate_cost_matrix(to_time_series(template), to_time_series(data))

        min_distance = None
        if self.min_stride_time not in (None, 0, 0.0):
            min_distance = self.min_stride_time * sampling_rate
        matches = self._find_matches(np.sqrt(self.acc_cost_mat_), self.threshold, min_distance)
        self.paths_ = self._find_multiple_paths(self.acc_cost_mat_, matches)
        self.costs_ = np.sqrt(self.acc_cost_mat_[-1, :][matches])

        return self.path_start_stops_

    @staticmethod
    def _calculate_cost_matrix(sub_sequence, long_sequence):
        return subsequence_cost_matrix(sub_sequence, long_sequence)

    def _interpolate_template(self, new_sampling_rate: float) -> np.ndarray:
        template = self.template
        if self.interpolate_template is True and new_sampling_rate != self.template_sampling_rate:
            template = resample(template, int(template.shape[0] * new_sampling_rate / self.template_sampling_rate))
        return template

    @staticmethod
    def _find_matches(acc_cost_mat: np.ndarray, max_cost: float, min_distance: float) -> np.ndarray:
        return find_local_minima_with_distance(acc_cost_mat[-1, :], max_cost=max_cost, distance=min_distance)

    @staticmethod
    def _find_multiple_paths(acc_cost_mat: np.ndarray, start_points: np.ndarray) -> List[np.ndarray]:
        # Note: This return the paths in non sorted order if run in parallel mode
        paths = []
        for i in range(len(start_points)):
            path = subsequence_path(acc_cost_mat, start_points[i])
            path_array = np.array(path)
            paths.append(path_array)
        return paths
