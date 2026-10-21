"""A version of BarthDTW that used local warping constrains by default."""

from typing import Literal

from joblib import Memory
from tpcp import CloneFactory

from gaitmap.utils._types import _Hashable
from gaitmap_mad.stride_segmentation.dtw._barth_dtw import BarthDtw
from gaitmap_mad.stride_segmentation.dtw._dtw_templates import BarthOriginalTemplate, BaseDtwTemplate


class ConstrainedBarthDtw(BarthDtw):
    """A version of BarthDtw that uses local warping constraints by default.

    This method is identical to :class:`~gaitmap.stride_segmentation.BarthDtw`, but uses local constraints for the
    template and the signal by default.
    This help to prevent issues, where only the region from TC to IC is mapped as the entire stride.
    For more information on this see the :ref:`example <example_constrained_barth_stride_segmentation>`.

    This exists as a separate class, so that users are aware, they are using a different method that might impact
    their results.

    This method is based on [1]_ and the modification of the constrained warping path is inspired by [2]_.

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
        For either peak-finding method, this parameter also sets the minimum distance between candidate endings.
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
        The rounded sample limit counts consecutive vertical steps. A run of L steps maps L + 1 template samples
        to one signal sample; the first matched cell consumes no step.
    max_signal_stretch_ms
        A local warping constraint for the DTW.
        It describes how many ms of the signal are allowed to be mapped to just a single datapoint of the template.
        The ms value will internally be converted to samples using the data sampling-rate.
        The rounded sample limit counts consecutive horizontal steps. A run of L steps maps L + 1 signal samples
        to one template sample; the first matched cell consumes no step.
    find_matches_method
        Select the method used to find matches in the cost function.

        - "min_under_thres"
            Matches the implementation used in the paper [1]_ to detect strides in foot mounted IMUs.
            In this case :func:`~gaitmap.stride_segmentation.base_dtw.find_matches_min_under_threshold` will be used as
            method.
        - "find_peaks"
            Historical interior-only minima and distance suppression via
            :func:`~gaitmap.stride_segmentation.find_matches_find_peaks`.
        - "find_peaks_with_boundaries" (default)
            Also considers minima and flat minima touching a recording edge, where a valid match can end.
            Boundary candidates are validated before competing for minimum distance. See
            :func:`~gaitmap.stride_segmentation.find_matches_find_peaks_with_boundaries` for details.
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
    The implementation changed after version 2.6 to correct the handling of direction/run-state constraints.
    Previously, it retained only the cheapest path at each cell, without distinguishing its arrival state.
    A path with a higher intermediate cost could be needed to complete the cheapest valid alignment,
    but was discarded in favour of a cheaper partial path that could not be extended without violating the limits.

    The historical backwards pass also followed neighbouring minimum costs without enforcing the run-state
    constraints. Unlike ordinary DTW, the two-dimensional cost matrix alone does not define a valid constrained
    path. Traceback needs the state information from the forward pass. The current implementation retains the
    cheapest arrival per direction/run state and traces its stored predecessors.

    In the `paired issue-93 benchmark`_, this correction reduced pooled F1 from 92.0274% to 91.9701% with unchanged
    defaults on 14 healthy participants' instep recordings. Numerical correctness does not guarantee better stride
    detection with existing parameters. For reproducibility, the previous calculation and traceback remain available
    as :class:`~gaitmap.stride_segmentation.IncorrectButBetterConstrainedBarthDtw`, which retains the known defects
    and is not recommended for new analyses.

    Both classes share the current boundary-aware peak selection by default. To also reproduce historical peak
    handling, explicitly set ``find_matches_method="find_peaks"`` when using the historical class.

    .. _paired issue-93 benchmark:
       https://github.com/mad-lab-fau/gaitmap/blob/564a431640a4c9399e44b735ebebdf28548dece1/docs/source/development/constrained_dtw_benchmark.rst

    .. [1] Barth, J., Oberndorfer, C., Kugler, P., Schuldhaus, D., Winkler, J., Klucken, J., & Eskofier, B. (2013).
       Subsequence dynamic time warping as a method for robust step segmentation using gyroscope signals of daily life
       activities. Proceedings of the Annual International Conference of the IEEE Engineering in Medicine and Biology
       Society, EMBS, 6744-6747. https://doi.org/10.1109/EMBC.2013.6611104
    .. [2] M. Müller, Fundamentals of Music Processing - Using Python and Jupyter Notebooks, 2nd ed. Springer Verlag,
       2021.


    See Also
    --------
    gaitmap.stride_segmentation.BarthDtw: For all details on the method

    """

    def __init__(
        self,
        template: BaseDtwTemplate | dict[_Hashable, BaseDtwTemplate] | None = CloneFactory(BarthOriginalTemplate()),
        resample_template: bool = True,
        find_matches_method: Literal["min_under_thres", "find_peaks", "find_peaks_with_boundaries"] = (
            "find_peaks_with_boundaries"
        ),
        max_cost: float | None = 4,
        min_match_length_s: float | None = 0.6,
        max_match_length_s: float | None = 3.0,
        max_template_stretch_ms: float | None = 120,
        max_signal_stretch_ms: float | None = 120,
        snap_to_min_win_ms: float | None = 300,
        snap_to_min_axis: str | None = "gyr_ml",
        conflict_resolution: bool = True,
        memory: Memory | None = None,
    ) -> None:
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
