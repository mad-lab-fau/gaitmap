"""A version of BarthDTW that used local warping constrains by default."""
from typing import Optional, Union, Dict

from joblib import Memory
from typing_extensions import Literal

from gaitmap.stride_segmentation.barth_dtw import BarthDtw
from gaitmap.stride_segmentation.dtw_templates import DtwTemplate, BarthOriginalTemplate
from gaitmap.utils._types import _Hashable


class ConstrainedBarthDtw(BarthDtw):
    """A version of BarthDtw that uses local warping constraints by default.

    This method is identical to :class:`~gaitmap.stride_segmentation.BarthDtw`, but uses local constraints for the
    template and the signal by default.
    This help to prevent issues, where only the region from TC to IC is mapped as the entire stride.
    For more information on this see the :ref:`example <example_constrained_barth_stride_segmentation>`.

    This exists as a separate class, so that users are aware, they are using a different method that might impact
    their results.

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
    .. [1] Barth, J., Oberndorfer, C., Kugler, P., Schuldhaus, D., Winkler, J., Klucken, J., & Eskofier, B. (2013).
       Subsequence dynamic time warping as a method for robust step segmentation using gyroscope signals of daily life
       activities. Proceedings of the Annual International Conference of the IEEE Engineering in Medicine and Biology
       Society, EMBS, 6744–6747. https://doi.org/10.1109/EMBC.2013.6611104

    See Also
    --------
    gaitmap.stride_segmentation.BarthDtw: For all details on the method

    """

    def __init__(
        self,
        template: Optional[Union[DtwTemplate, Dict[_Hashable, DtwTemplate]]] = BarthOriginalTemplate(),
        resample_template: bool = True,
        find_matches_method: Literal["min_under_thres", "find_peaks"] = "find_peaks",
        max_cost: Optional[float] = 4,
        min_match_length_s: Optional[float] = 0.6,
        max_match_length_s: Optional[float] = 3.0,
        max_template_stretch_ms: Optional[float] = 120,
        max_signal_stretch_ms: Optional[float] = 120,
        snap_to_min_win_ms: Optional[float] = 300,
        snap_to_min_axis: Optional[str] = "gyr_ml",
        conflict_resolution: bool = True,
        memory: Optional[Memory] = None,
    ):
        super().__init__(
            template=template,
            max_cost=max_cost,
            min_match_length_s=min_match_length_s,
            max_match_length_s=max_match_length_s,
            max_template_stretch_ms=max_template_stretch_ms,
            max_signal_stretch_ms=max_signal_stretch_ms,
            resample_template=resample_template,
            find_matches_method=find_matches_method,
            snap_to_min_win_ms=snap_to_min_win_ms,
            snap_to_min_axis=snap_to_min_axis,
            conflict_resolution=conflict_resolution,
            memory=memory,
        )
