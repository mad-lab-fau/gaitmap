"""The msDTW based stride segmentation algorithm by Barth et al 2013."""
from typing import Optional, Union, Dict, List, Tuple

import numpy as np
import pandas as pd
from typing_extensions import Literal

from gaitmap.stride_segmentation.base_dtw import BaseDtw
from gaitmap.stride_segmentation.dtw_templates.templates import DtwTemplate, BarthOriginalTemplate
from gaitmap.utils.array_handling import find_minima_in_radius


class BarthDtw(BaseDtw):
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
        By default, the :class:`~gaitmap.stride_segmentation.dtw_templates.templates.BarthOriginalTemplate` is used
        with default settings.
    resample_template
        If `True` the template will be resampled to match the sampling rate of the data.
        This requires a valid value for `template.sampling_rate_hz` value.
    max_cost
        The maximal allowed cost to find potential match in the cost function.
        Its usage depends on the exact `find_matches_method` used.
        Refer to the specific function to learn more about this.
        The default value should work well with healthy gait (with the default template).
    min_match_length_s
        The minimal length of a sequence in seconds to be still considered a stride.
        This exclusion is performed as a post-processing step after the matching.
        If "find_peaks" is selected as `find_matches_method`, the parameter is additionally used in the detection of
        matches directly.
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
        However, this assumes that the start and the end of each match is marked by a clear minima in one axis of the
        raw data.
        If you are using a template that does not assume this, this post-processing step might lead to unexpected
        results and you should deactivate it in such a case by setting `snap_to_min_win_ms` to `None`.

    TODO: Add additional details about the use of DTW for stride segmentation

    .. [1] Barth, J., Oberndorfer, C., Kugler, P., Schuldhaus, D., Winkler, J., Klucken, J., & Eskofier, B. (2013).
       Subsequence dynamic time warping as a method for robust step segmentation using gyroscope signals of daily life
       activities. Proceedings of the Annual International Conference of the IEEE Engineering in Medicine and Biology
       Society, EMBS, 6744–6747. https://doi.org/10.1109/EMBC.2013.6611104

    """

    snap_to_min_win_ms: Optional[float]
    snap_to_min_axis: Optional[str]

    def __init__(
        self,
        template: Optional[Union[DtwTemplate, Dict[str, DtwTemplate]]] = BarthOriginalTemplate(),
        resample_template: bool = True,
        find_matches_method: Literal["min_under_thres", "find_peaks"] = "find_peaks",
        max_cost: Optional[float] = 2000.0,
        min_match_length_s: Optional[float] = 0.6,
        snap_to_min_win_ms: Optional[float] = 100,
        snap_to_min_axis: Optional[str] = "gyr_ml",
    ):
        self.snap_to_min_win_ms = snap_to_min_win_ms
        self.snap_to_min_axis = snap_to_min_axis
        super().__init__(
            template=template,
            max_cost=max_cost,
            min_match_length_s=min_match_length_s,
            resample_template=resample_template,
            find_matches_method=find_matches_method,
        )

    @property
    def stride_list_(self) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Return start and end of each match as pd.DataFrame."""
        start_ends = self.matches_start_end_
        if isinstance(start_ends, dict):
            return {k: self._format_stride_list(v) for k, v in start_ends.items()}
        return self._format_stride_list(start_ends)

    @staticmethod
    def _format_stride_list(array: np.ndarray) -> pd.DataFrame:
        if len(array) == 0:
            array = None
        as_df = pd.DataFrame(array, columns=["start", "end"])
        # Add the s_id
        as_df["s_id"] = as_df.index
        return as_df

    def _postprocess_matches(self, data, matches_start_end: np.ndarray, paths: List) -> Tuple[np.ndarray, List]:
        matches_start_end, paths = super()._postprocess_matches(
            data=data, matches_start_end=matches_start_end, paths=paths
        )
        # Apply snap to minimum
        if self.snap_to_min_win_ms:
            # Find the closest minimum for each start and stop value
            matches_start_end = find_minima_in_radius(
                data[self.snap_to_min_axis].to_numpy(),
                matches_start_end.flatten(),
                int(self.snap_to_min_win_ms * self.sampling_rate_hz / 1000) // 2,
            ).reshape(matches_start_end.shape)
        return matches_start_end, paths