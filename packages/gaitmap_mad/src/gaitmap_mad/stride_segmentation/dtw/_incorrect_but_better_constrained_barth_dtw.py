"""Preserve historical constrained-DTW costs and traceback."""

import numpy as np
from numba import njit

from gaitmap_mad.stride_segmentation.dtw._constrained_barth_dtw import ConstrainedBarthDtw
from gaitmap_mad.stride_segmentation.dtw._vendored_tslearn import _local_squared_dist


class IncorrectButBetterConstrainedBarthDtw(ConstrainedBarthDtw):
    """Reproduce the constrained-DTW algorithm used before the issue-93 correction.

    See the notes in :class:`~gaitmap.stride_segmentation.ConstrainedBarthDtw` for details on the issue
    and its correction.

    Parameters
    ----------
    template
        Matching template, defaulting to :class:`~gaitmap.stride_segmentation.BarthOriginalTemplate`.
    resample_template
        Resample the template to the signal sampling rate when True.
    find_matches_method
        Defaults to "find_peaks_with_boundaries", as in the corrected class.
        Select "find_peaks" to also reproduce historical interior-only peak selection,
        or "min_under_thres" for minima in below-threshold regions.
    max_cost
        Maximum cost for a candidate match; None disables the threshold for either peak-finding method.
    min_match_length_s
        Minimum match duration in seconds, also used as peak distance for either peak-finding method.
    max_match_length_s
        Maximum match duration in seconds.
    max_template_stretch_ms
        Historical vertical run limit in milliseconds, rounded to template samples.
        None disables this limit. Returned paths may violate it.
    max_signal_stretch_ms
        Historical horizontal run limit in milliseconds, rounded to signal samples.
        None disables this limit. Returned paths may violate it.
    snap_to_min_win_ms
        Window in milliseconds for snapping stride borders to minima; None disables snapping.
    snap_to_min_axis
        Signal column used for snapping, defaulting to "gyr_ml".
    conflict_resolution
        Resolve overlapping stride candidates and duplicate starts when True.
    memory
        Optional joblib.Memory for caching cost calculations.

    Attributes
    ----------
    stride_list_
        Detected strides as a DataFrame, or a dictionary per sensor.
    matches_start_end_
        Detected start and exclusive end sample indices, optionally after snapping.
    matches_start_end_original_
        Start and exclusive end sample indices before snapping.
    costs_
        Reported cost of each match; it can differ from the cost along ``paths_``.
    acc_cost_mat_
        Historical accumulated cost matrix, or a dictionary per sensor.
    cost_function_
        Square root of the accumulated matrix's final row.
    paths_
        Historical traceback paths, which can violate the requested limits.

    Other Parameters
    ----------------
    data
        Signal supplied to segment().
    sampling_rate_hz
        Sampling rate supplied to segment().

    See Also
    --------
    gaitmap.stride_segmentation.ConstrainedBarthDtw : Corrected constrained DTW.

    """

    def _select_cost_matrix_method(self, max_template_stretch, max_signal_stretch):
        if max_template_stretch == np.inf and max_signal_stretch == np.inf:
            return super()._select_cost_matrix_method(max_template_stretch, max_signal_stretch)
        # An ordinary matrix selects historical traceback; peak selection remains independent.
        return _historical_cost_matrix, {
            "max_subseq_steps": max_template_stretch,
            "max_longseq_steps": max_signal_stretch,
        }


def _historical_cost_matrix(subseq, longseq, max_subseq_steps, max_longseq_steps):
    return _historical_subsequence_cost_matrix(subseq, longseq, max_subseq_steps, max_longseq_steps)[..., 0]


@njit(cache=True)
def _historical_subsequence_cost_matrix(subseq, longseq, max_subseq_steps, max_longseq_steps):
    # We consider longseq directions as negative and subseq directions as positive values
    max_longseq_steps = -max_longseq_steps
    l1 = subseq.shape[0]
    l2 = longseq.shape[0]
    cum_sum = np.full((l1 + 1, l2 + 1, 2), np.inf)
    cum_sum[0, :] = 0.0
    # All counter values are set to 0
    cum_sum[:, :, 1] = 0

    for i in range(l1):
        for j in range(l2):
            cum_sum[i + 1, j + 1, 0] = _local_squared_dist(subseq[i], longseq[j])
            vals = np.empty((3, 2))
            shifts = [(1, 0), (0, 1), (0, 0)]
            # Generate all possible outcomes and check if any of them violate the constrains.
            for index, shift in enumerate(shifts):
                vals[index, :] = cum_sum[i + shift[0], j + shift[1]]
            # Check if either the number vertical or horizontal step exceed the threshold.
            # In this case set the distance to infinite, so that this direction can not be picked.
            # vertical step
            if vals[0, 1] <= max_longseq_steps:
                vals[0, 0] = np.inf
            if vals[1, 1] >= max_subseq_steps:
                vals[1, 0] = np.inf
            smallest_cost = np.argmin(vals[:, 0])
            # update the step counter based on what step is taken (smallest cost)
            # We only need to update the step counter, if the smallest cost wasn't the diagonal step (2)
            if smallest_cost != 2:
                step = -1 if smallest_cost == 0 else 1
                current_counter = vals[smallest_cost, 1]
                if np.sign(step) != np.sign(current_counter):
                    # The step changes direction from horizontal to vertical
                    # This means we reset the counter
                    current_counter = 0
                cum_sum[i + 1, j + 1, 1] = current_counter + step
            cum_sum[i + 1, j + 1, 0] += vals[smallest_cost, 0]
    return cum_sum[1:, 1:]
