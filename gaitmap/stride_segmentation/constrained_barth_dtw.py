from typing import Optional, Union, Dict

import numpy as np
from numba import njit
from tslearn.metrics import _local_squared_dist
from tslearn.utils import to_time_series
from typing_extensions import Literal

from gaitmap.stride_segmentation.barth_dtw import BarthDtw
from gaitmap.stride_segmentation.dtw_templates import DtwTemplate, BarthOriginalTemplate


class ConstrainedDtw(BarthDtw):
    def __init__(
        self,
        template: Optional[Union[DtwTemplate, Dict[str, DtwTemplate]]] = BarthOriginalTemplate(),
        resample_template: bool = True,
        find_matches_method: Literal["min_under_thres", "find_peaks"] = "find_peaks",
        max_cost: Optional[float] = 4.0,
        min_match_length_s: Optional[float] = 0.6,
        max_match_length_s: Optional[float] = 3.0,
        max_template_stretch: Optional[int] = 20,
        max_signal_strech: Optional[int] = 30,
        snap_to_min_win_ms: Optional[float] = 300,
        snap_to_min_axis: Optional[str] = "gyr_ml",
        conflict_resolution: bool = True,
    ):
        self.max_template_stretch = max_template_stretch
        self.max_signal_strech = max_signal_strech
        super().__init__(
            template=template,
            max_cost=max_cost,
            min_match_length_s=min_match_length_s,
            max_match_length_s=max_match_length_s,
            resample_template=resample_template,
            find_matches_method=find_matches_method,
            snap_to_min_win_ms=snap_to_min_win_ms,
            snap_to_min_axis=snap_to_min_axis,
            conflict_resolution=conflict_resolution,
        )

    def _calculate_cost_matrix(self, template, matching_data):
        cost_matrix = subsequence_cost_matrix_with_constrains(
            to_time_series(template), to_time_series(matching_data), self.max_template_stretch, self.max_signal_strech
        )
        return cost_matrix[..., 0]


@njit()
def subsequence_cost_matrix_with_constrains(subseq, longseq, max_subseq_steps, max_longseq_steps):
    l1 = subseq.shape[0]
    l2 = longseq.shape[0]
    cum_sum = np.full((l1 + 1, l2 + 1, 3), np.inf)
    cum_sum[0, :] = 0.0
    cum_sum[0:] = 0

    for i in range(l1):
        for j in range(l2):
            cum_sum[i + 1, j + 1, 0] = _local_squared_dist(subseq[i], longseq[j])
            vals = np.empty((3, 3))
            shifts = [(1, 0), (0, 1), (0, 0)]
            for index in range(3):
                shift = shifts[index]
                vals[index, :] = cum_sum[i + shift[0], j + shift[1]]
                if index == 0 and vals[index, 1] >= max_subseq_steps:
                    vals[index, 0] = np.inf
                elif index == 1 and vals[index, 2] >= max_longseq_steps:
                    vals[index, 0] = np.inf

            smallest_cost = np.argmin(vals[:, 0])
            if smallest_cost == 0:
                cum_sum[i + 1, j + 1, 1] = vals[0, 1] + 1
                cum_sum[i + 1, j + 1, 2] = 0
            elif smallest_cost == 1:
                cum_sum[i + 1, j + 1, 2] += vals[1, 2] + 1
                cum_sum[i + 1, j + 1, 1] = 0
            cum_sum[i + 1, j + 1, 0] += vals[smallest_cost, 0]
    return cum_sum[1:, 1:]
