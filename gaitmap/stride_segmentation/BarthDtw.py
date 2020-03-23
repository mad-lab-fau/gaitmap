from typing import Optional, Sequence, List

import numpy as np
from scipy.signal import resample
from tslearn.metrics import subsequence_cost_matrix, subsequence_path
from tslearn.utils import to_time_series
from typing_extensions import Literal

from gaitmap.base import BaseStrideSegmentation
from gaitmap.stride_segmentation.utils import find_local_minima_with_distance, find_local_minima_below_threshold


def find_matches_find_peaks(acc_cost_mat: np.ndarray, max_cost: float, min_distance: float) -> np.ndarray:
    return find_local_minima_with_distance(np.sqrt(acc_cost_mat[-1, :]), threshold=max_cost, distance=min_distance)


def find_matches_original(acc_cost_mat: np.ndarray, max_cost: float, **_) -> np.ndarray:
    return find_local_minima_below_threshold(np.sqrt(acc_cost_mat[-1, :]), threshold=max_cost)


class BarthDtw(BaseStrideSegmentation):
    """Segment strides using a single stride template and Dynamic Time Warping."""

    template: np.ndarray
    template_sampling_rate: float
    interpolate_template: bool
    threshold: Optional[float]
    min_stride_time_s: float
    find_matches_method: Literal["original", "find_peaks"]

    acc_cost_mat_: Optional[np.ndarray] = None
    paths_: Optional[Sequence[Sequence[tuple]]] = None
    costs_: Optional[Sequence[float]] = None

    data: np.ndarray
    sampling_rate: float

    _allowed_methods_map = {"original": find_matches_original, "find_peaks": find_matches_find_peaks}

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
        interpolate_template: bool = True,
        find_matches_method: Literal["original", "find_peaks"] = "original",
        min_stride_time_s: Optional[float] = 0.6,
    ):
        self.template = template
        self.template_sampling_rate = template_sampling_rate
        self.threshold = threshold
        self.min_stride_time_s = min_stride_time_s
        self.interpolate_template = interpolate_template
        self.find_matches_method = find_matches_method

    def segment(self, data: np.ndarray, sampling_rate: float, **kwargs) -> List[List[int]]:
        self.data = data
        self.sampling_rate = sampling_rate

        # Validate and transform inputs
        template = self._interpolate_template(sampling_rate)
        min_distance = None
        if self.min_stride_time_s not in (None, 0, 0.0):
            min_distance = self.min_stride_time_s * sampling_rate
        find_matches_method = self._allowed_methods_map.get(self.find_matches_method, None)
        if not find_matches_method:
            raise ValueError(
                'Invalid value for "find_matches_method". Must be one of {}'.format(
                    list(self._allowed_methods_map.keys())
                )
            )

        # Calculate cost matrix
        self.acc_cost_mat_ = subsequence_cost_matrix(to_time_series(template), to_time_series(data))

        matches = find_matches_method(
            acc_cost_mat=self.acc_cost_mat_, max_cost=self.threshold, min_distance=min_distance
        )
        self.paths_ = self._find_multiple_paths(self.acc_cost_mat_, matches)
        self.costs_ = np.sqrt(self.acc_cost_mat_[-1, :][matches])

        return self.path_start_stops_

    def _interpolate_template(self, new_sampling_rate: float) -> np.ndarray:
        template = self.template
        if self.interpolate_template is True and new_sampling_rate != self.template_sampling_rate:
            template = resample(template, int(template.shape[0] * new_sampling_rate / self.template_sampling_rate))
        return template

    @staticmethod
    def _find_multiple_paths(acc_cost_mat: np.ndarray, start_points: np.ndarray) -> List[np.ndarray]:
        # Note: This return the paths in non sorted order if run in parallel mode
        paths = []
        for i in range(len(start_points)):
            path = subsequence_path(acc_cost_mat, start_points[i])
            path_array = np.array(path)
            paths.append(path_array)
        return paths
