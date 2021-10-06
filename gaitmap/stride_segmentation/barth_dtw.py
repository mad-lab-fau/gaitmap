"""The msDTW based stride segmentation algorithm by Barth et al 2013."""
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Memory
from typing_extensions import Literal

from gaitmap.base import BaseStrideSegmentation
from gaitmap.stride_segmentation.base_dtw import BaseDtw
from gaitmap.stride_segmentation.dtw_templates.templates import BarthOriginalTemplate, DtwTemplate
from gaitmap.utils._types import _Hashable
from gaitmap.utils.array_handling import find_extrema_in_radius
from gaitmap.utils.datatype_helper import StrideList


class BarthDtw(BaseDtw, BaseStrideSegmentation):
    """Segment strides using a single stride template and Dynamic Time Warping.

    BarthDtw uses a manually created template of an IMU stride to find multiple occurrences of similar signals in a
    continuous data stream.
    The method is not limited to a single sensor-axis or sensor, but can use any number of dimensions of the provided
    input signal simultaneously.
    For more details refer to the `Notes` section.

    Parameters
    ----------
    template
        The template used for matching.
        The required data type and shape depends on the use case.
        For more details see :class:`~gaitmap.stride_segmentation.BaseDtw`.
        By default, the :class:`~gaitmap.stride_segmentation.BarthOriginalTemplate` is used
        with default settings.
    resample_template
        If `True` the template will be resampled to match the sampling rate of the data.
        This requires a valid value for `template.sampling_rate_hz` value.
    max_cost
        The maximal allowed cost to find potential match in the cost function.
        Note that the cost is roughly calculated as: `sqrt(|template - data/template.scaling|)`.
        Its usage depends on the exact `find_matches_method` used.
        Refer to the specific function to learn more about this.
        The default value should work well with healthy gait (with the default template).
    min_match_length_s
        The minimal length of a sequence in seconds to be considered a stride.
        Matches that result in shorter sequences, will be ignored.
        In general, this exclusion is performed as a post-processing step after the matching.
        If "find_peaks" is selected as `find_matches_method`, the parameter is additionally used in the detection of
        matches directly.
    max_match_length_s
        The maximal length of a sequence in seconds to be considered a stride.
        Matches that result in longer sequences will be ignored.
        This exclusion is performed as a post-processing step after the matching.
    max_template_stretch_ms
        A local warping constraint for the DTW.
        It describes how many ms of the template are allowed to be mapped to just a single datapoint of the signal.
        The ms value will internally be converted to samples using the template sampling-rate (or the signal
        sampling-rate, if `resample_template=True`).
        If no template sampling-rate is provided, this constrain can not be used.
    max_signal_stretch_ms
        A local warping constraint for the DTW.
        It describes how many ms of the signal are allowed to be mapped to just a single datapoint of the template.
        The ms value will internally be converted to samples using the data sampling-rate
    find_matches_method
        Select the method used to find matches in the cost function.

        - "min_under_thres"
            Matches the implementation used in the paper [1]_ to detect strides in foot mounted IMUs.
            In this case :func:`~gaitmap.stride_segmentation.base_dtw.find_matches_find_peaks` will be used as method.
        - "find_peaks"
            Uses :func:`~scipy.signal.find_peaks` with additional constraints to find stride candidates.
            In this case :func:`~gaitmap.stride_segmentation.base_dtw.find_matches_min_under_threshold` will be used as
            method.
    snap_to_min_win_ms
        The size of the window in ms used to search local minima during the post processing of the stride borders.
        If this is set to None, this postprocessing step is skipped.
        Refer to the Notes section for more details.
    snap_to_min_axis
        The axis of the data used to search for minima during the processing of the stride borders.
        The axis label must match one of the axis label in the data.
        Refer to the Notes section for more details.
    conflict_resolution
        This enables a set of checks that handle cases where stride matches overlap with other strides.
        The following steps will be performed:

        - If multiple matches have the same start point, only the match with the lowest cost will be kept.
    memory
        An optional `joblib.Memory` object that can be provided to cache the creation of cost matrizes and the peak
        detection.

    Attributes
    ----------
    stride_list_ : A stride list or dictionary with such values
        The same output as `matches_start_end_`, but as properly formatted pandas DataFrame that can be used as input to
        other algorithms.
        If `snap_to_min_window_ms` is not `None`, the start and end value might not match to the output of `paths_`.
        Refer to `matches_start_end_original_` for the unmodified start and end values.
    matches_start_end_ : 2D array of shape (n_detected_strides x 2) or dictionary with such values
        The start (column 1) and end (column 2) of each detected stride.
    costs_ : List of length n_detected_strides or dictionary with such values
        The cost value associated with each stride.
    acc_cost_mat_ : array with the shapes (length_template x length_data) or dictionary with such values
        The accumulated cost matrix of the DTW. The last row represents the cost function.
    cost_function_ : 1D array with the same length as the data or dictionary with such values
        The final cost function calculated as the square root of the last row of the accumulated cost matrix.
    paths_ : list of arrays with length n_detected_strides or dictionary with such values
        The full path through the cost matrix of each detected stride.
    matches_start_end_original_ : 2D array of shape (n_detected_strides x 2) or dictionary with such values
        Identical to `matches_start_end_` if `snap_to_min_window_ms` is equal to `None`.
        Otherwise, it return the start and end values before the sanpping is applied.

    Other Parameters
    ----------------
    data
        The data passed to the `segment` method.
    sampling_rate_hz
        The sampling rate of the data

    Notes
    -----
    Post Processing
        This algorithm uses an optional post-processing step that "snaps" the stride borders to the closest local
        minimum in the raw data.
        This helps to align the end of one stride with the start of the next stride (which is a requirement for certain
        event detection algorithms) and resolve small overlaps between neighboring strides.
        However, this assumes that the start and the end of each match is marked by a clear minimum in one axis of the
        raw data.
        If you are using a template that does not assume this, this post-processing step might lead to unexpected
        results and you should deactivate it in such a case by setting `snap_to_min_win_ms` to `None`.
        Note, that snapping only works if both the data and the template are based on pandas DataFrames.
    Template
        The :class:`~gaitmap.stride_segmentation.BarthOriginalTemplate` covers a gait signal starting from the minimum
        in `gyr_ml` before a terminal contact until the same minimum of the next gait cycle.
        It is advisable that any custom template starts and ends with a clear peak as well, as this improves the
        matching performance in the border regions.
    Initiation and Termination Strides
        Be aware that initiation and termination strides can usually not be matched with the same template as regular
        strides.
        However, depending on the gait of the subject and the chosen `max_cost` parameters it might however happen
        that some of them are matched on occasion.
        If this is an issue for your analysis, you should try to develop further post-processing steps to exclude these
        strides as part of your pipeline.
    Differences to implementation by Barth et al [1]_
        In the original paper, the distance matrix for each of the selected axis was calculated individually and then
        summed up to create the final matrix.
        The resulting distance function is therefore the Manhatten distance between two samples.
        This implementation calculates only a single distance matrix using the Euclidian distance.
        The Euclidian distance is always a little smaller than the Manhatten distance, but captures the same similarity.

    .. [1] Barth, J., Oberndorfer, C., Kugler, P., Schuldhaus, D., Winkler, J., Klucken, J., & Eskofier, B. (2013).
       Subsequence dynamic time warping as a method for robust step segmentation using gyroscope signals of daily life
       activities. Proceedings of the Annual International Conference of the IEEE Engineering in Medicine and Biology
       Society, EMBS, 6744–6747. https://doi.org/10.1109/EMBC.2013.6611104


    See Also
    --------
    gaitmap.stride_segmentation.ConstrainedBarthDtw: Version with defaults for the local constrains
    gaitmap.stride_segmentation.BaseDtw: Dtw base class with more details on the method

    """

    snap_to_min_win_ms: Optional[float]
    snap_to_min_axis: Optional[str]
    conflict_resolution: bool

    def __init__(
        self,
        template: Optional[Union[DtwTemplate, Dict[_Hashable, DtwTemplate]]] = BarthOriginalTemplate(),
        resample_template: bool = True,
        find_matches_method: Literal["min_under_thres", "find_peaks"] = "find_peaks",
        max_cost: Optional[float] = 4.0,
        min_match_length_s: Optional[float] = 0.6,
        max_match_length_s: Optional[float] = 3.0,
        max_template_stretch_ms: Optional[float] = None,
        max_signal_stretch_ms: Optional[float] = None,
        snap_to_min_win_ms: Optional[float] = 300,
        snap_to_min_axis: Optional[str] = "gyr_ml",
        conflict_resolution: bool = True,
        memory: Optional[Memory] = None,
    ):
        self.snap_to_min_win_ms = snap_to_min_win_ms
        self.snap_to_min_axis = snap_to_min_axis
        self.conflict_resolution = conflict_resolution
        super().__init__(
            template=template,
            max_cost=max_cost,
            min_match_length_s=min_match_length_s,
            max_match_length_s=max_match_length_s,
            max_template_stretch_ms=max_template_stretch_ms,
            max_signal_stretch_ms=max_signal_stretch_ms,
            resample_template=resample_template,
            find_matches_method=find_matches_method,
            memory=memory,
        )

    @property
    def stride_list_(self) -> StrideList:
        """Return start and end of each match as pd.DataFrame."""
        start_ends = self.matches_start_end_
        if isinstance(start_ends, dict):
            return {k: self._format_stride_list(v) for k, v in start_ends.items()}
        return self._format_stride_list(start_ends)

    @stride_list_.setter
    def stride_list_(self, arg: StrideList):  # noqa: no-self-use
        """Fake setter for the stride list.

        This is required to be type compatible with the base class.
        """
        raise ValueError("The argument `stride_list_` is readonly.")

    @staticmethod
    def _format_stride_list(array: np.ndarray) -> pd.DataFrame:
        tmp: Optional[np.ndarray] = array
        if len(array) == 0:
            tmp = None
        as_df = pd.DataFrame(tmp, columns=["start", "end"])
        # Add the s_id
        as_df.index.name = "s_id"
        return as_df

    def _postprocess_matches(
        self,
        data,
        paths: List,
        cost: np.ndarray,
        matches_start_end: np.ndarray,
        acc_cost_mat: np.ndarray,
        to_keep: np.ndarray,
        memory: Memory,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Apply snap to minimum
        if self.snap_to_min_win_ms:
            # Find the closest minimum for each start and stop value
            flattened_matches = matches_start_end.flatten()
            # Because the actual end values are exclusive, we need to handle the case were the stride ends inclusive the
            # last value
            edge_case_stride = flattened_matches == len(data)
            flattened_matches[edge_case_stride] -= 1
            flattened_matches = find_extrema_in_radius(
                data[self.snap_to_min_axis].to_numpy(),
                flattened_matches,
                int(self.snap_to_min_win_ms * self.sampling_rate_hz / 1000) // 2,
            )
            # All strides that were inclusive with the last sample and didn't change the sample will be changed back
            # to be inclusive.
            # Strides that were "snapped" to the last sample are exclusive the last sample.
            # Their remains an edge case were a stride that was inclusive the last sample was correctly snapped to be
            # exclusive and is then updated to be inclusive again in the following line.
            # However, this is not worth handling.
            flattened_matches[edge_case_stride & (flattened_matches == len(data) - 1)] += 1
            matches_start_end = flattened_matches.reshape(matches_start_end.shape)

        # Apply any postprocessing steps of the parent class.
        # This is done after the snapping, as the snapping might modify the stride time.
        matches_start_end, to_keep = super()._postprocess_matches(
            data=data,
            matches_start_end=matches_start_end,
            paths=paths,
            cost=cost,
            to_keep=to_keep,
            acc_cost_mat=acc_cost_mat,
            memory=memory,
        )

        # Resolve strides that have the same start point
        # In case multiple strides have the same start point (after snapping) only take the one with lower cost
        # Only run calcs for strides that are not excluded already
        valid_indices = np.where(to_keep)[0]

        if self.conflict_resolution and len(valid_indices) > 0:
            valid_matches_start_end = matches_start_end[valid_indices]
            # Just to be sure sort based on the start value
            sorted_indices = np.argsort(valid_matches_start_end[:, 0])
            valid_indices = valid_indices[sorted_indices]
            valid_matches_start_end = valid_matches_start_end[sorted_indices]

            starts = valid_matches_start_end[:, 0].astype(float)
            cost_per_valid_stride = cost[valid_indices]
            # get groups of strides with the same start value
            strides_with_same_start = np.diff(starts, prepend=np.inf) == 0
            starts[~strides_with_same_start] = np.nan
            groups = np.ma.clump_unmasked(np.ma.masked_invalid(starts))
            for s in groups:
                # For each group find the stride with the lowest original cost
                indices = np.arange(s.start - 1, s.stop)
                keep = np.argmin(cost_per_valid_stride[indices])
                # Remove all strides except the one with the lowest cost.
                to_keep[valid_indices[indices]] = False
                to_keep[valid_indices[indices[keep]]] = True

        return matches_start_end, to_keep

    def _post_postprocess_check(self, matches_start_end):
        super()._post_postprocess_check(matches_start_end)
        # Check if there are still overlapping strides
        if np.any(np.diff(matches_start_end.flatten()) < 0):
            warnings.warn(
                "There are still overlapping strides left after postprocessing. Please manually review the results."
            )
