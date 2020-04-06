"""The msDTW based stride segmentation algorithm by Barth et al 2013."""
from typing import Optional, Union, Dict

import numpy as np
import pandas as pd
from typing_extensions import Literal

from gaitmap.base import BaseType
from gaitmap.stride_segmentation.base_dtw import BaseDtw
from gaitmap.stride_segmentation.dtw_templates.templates import DtwTemplate
from gaitmap.utils.dataset_helper import Dataset


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
        For more details see :class:`BaseDtw <gaitmap.stride_segmentation.base_dtw.BaseDtw>`.
    resample_template
        If `True` the template will be resampled to match the sampling rate of the data.
        This requires a valid value for `template.sampling_rate_hz` value.
    max_cost
        The maximal allowed cost to find potential match in the cost function.
        Its usage depends on the exact `find_matches_method` used.
        Refer to the specific funtion to learn more about this.
    min_stride_time_s
        The minimal length of a sequence in seconds to be still considered a stride.
        This is just a more convenient way to set `min_match_length`.
        If both are provided `min_stride_time_s` is used and converted into samples based on the data sampling rate.
    min_match_length
        The minimal length of a sequence in samples to be considered a match.
        Matches that result in shorter sequences, will be ignored.
        At the moment this is only used if "find_peaks" is selected as `find_matches_method`.
    find_matches_method
        Select the method used to find matches in the cost function.

        - "min_under_thres"
            Matches the implementation used in the paper [1]_ to detect strides in foot mounted IMUs.
            In this case :py:func:`.find_matches_min_under_threshold` will be used as method.
        - "find_peaks"
            Uses :func:`scipy.signal.find_peaks` with additional constraints to find stride candidates.
            In this case :py:func:`.find_matches_find_peaks` will be used as method.

    Attributes
    ----------
    stride_list_ : A stride list or dictionary with such values
        The same output as `matches_start_end_`, but as properly formatted pandas DataFrame that can be used as input to
        other algorithms.
    matches_start_end_ : 2D array of shape (n_detected_strides x 2) or dictionary with such values
        The start (column 1) and stop (column 2) of each detected stride.
    costs_ : List of length n_detected_strides or dictionary with such values
        The cost value associated with each stride.
    acc_cost_mat_ : array with the shapes (length_template x length_data) or dictionary with such values
        The accumulated cost matrix of the DTW. The last row represents the cost function.
    cost_function_ : 1D array with the same length as the data or dictionary with such values
        The final cost function calculated as the square root of the last row of the accumulated cost matrix.
    paths_ : list of arrays with length n_detected_strides or dictionary with such values
        The full path through the cost matrix of each detected stride.

    Other Parameters
    ----------------
    data
        The data passed to the `segment` method.
    sampling_rate_hz
        The sampling rate of the data

    Notes
    -----
    TODO: Add additional details about the use of DTW for stride segmentation

    .. [1] Barth, J., Oberndorfer, C., Kugler, P., Schuldhaus, D., Winkler, J., Klucken, J., & Eskofier, B. (2013).
       Subsequence dynamic time warping as a method for robust step segmentation using gyroscope signals of daily life
       activities. Proceedings of the Annual International Conference of the IEEE Engineering in Medicine and Biology
       Society, EMBS, 6744–6747. https://doi.org/10.1109/EMBC.2013.6611104

    """

    min_stride_time_s: Optional[float]

    def __init__(
        self,
        template: Optional[Union[DtwTemplate, Dict[str, DtwTemplate]]] = None,
        resample_template: bool = True,
        find_matches_method: Literal["min_under_thres", "find_peaks"] = "find_peaks",
        max_cost: Optional[float] = None,
        min_stride_time_s: Optional[float] = 0.6,
        min_match_length: Optional[float] = None,
    ):
        self.min_stride_time_s = min_stride_time_s
        super().__init__(
            template=template,
            max_cost=max_cost,
            min_match_length=min_match_length,
            resample_template=resample_template,
            find_matches_method=find_matches_method,
        )

    def segment(self: BaseType, data: Union[np.ndarray, Dataset], sampling_rate_hz: float, **_) -> BaseType:
        if self.min_stride_time_s not in (None, 0, 0.0):
            self.min_match_length = self.min_stride_time_s * sampling_rate_hz
        return super().segment(data=data, sampling_rate_hz=sampling_rate_hz)

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
        return pd.DataFrame(array, columns=["start", "end"])
