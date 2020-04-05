"""A implementation of a sDTW that can be used independent of the context of Stride Segmentation."""

from typing import Optional, Sequence, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import resample
from tslearn.metrics import subsequence_cost_matrix, subsequence_path
from tslearn.utils import to_time_series
from typing_extensions import Literal

from gaitmap.base import BaseStrideSegmentation, BaseType
from gaitmap.stride_segmentation.dtw_templates import DtwTemplate
from gaitmap.stride_segmentation.utils import find_local_minima_with_distance, find_local_minima_below_threshold
from gaitmap.utils.dataset_helper import Dataset


def find_matches_find_peaks(acc_cost_mat: np.ndarray, max_cost: float, min_distance: float) -> np.ndarray:
    """Find matches in the accumulated cost matrix using :func:`scipy.signal.find_peaks`.

    Parameters
    ----------
    acc_cost_mat
        Accumulated cost matrix as derived from a DTW
    max_cost
        The max_cost is used as `height` value for the `find_peaks` function.
    min_distance
        The min distance in samples. This is used as the `distance` value for the `find_peaks` function.

    Returns
    -------
    list_of_matches
        A list of indices marking the end of a potential stride.

    See Also
    --------
    gaitmap.stride_segmentation.utils.find_local_minima_with_distance: Details on the function call to `find_peaks`.
    scipy.signal.find_peaks: The actual `find_peaks` method.

    """
    return find_local_minima_with_distance(np.sqrt(acc_cost_mat[-1, :]), threshold=max_cost, distance=min_distance)


def find_matches_min_under_threshold(acc_cost_mat: np.ndarray, max_cost: float, **_) -> np.ndarray:
    """Find matches in the accumulated cost matrix by searching for minima in sections enclosed by the `max_cost`.

    Parameters
    ----------
    acc_cost_mat
        Accumulated cost matrix as derived from a DTW
    max_cost
        The max_cost is used to cut the signal into enclosed segments.
        More details at :py:func:`find_local_minima_below_threshold
        <gaitmap.stride_segmentation.utils.find_local_minima_below_threshold>`.

    Returns
    -------
    list_of_matches
        A list of indices marking the end of a potential stride.

    See Also
    --------
    gaitmap.stride_segmentation.utils.find_local_minima_below_threshold: Implementation details.

    """
    return find_local_minima_below_threshold(np.sqrt(acc_cost_mat[-1, :]), threshold=max_cost)


class BaseDtw(BaseStrideSegmentation):
    """A basic implementation of subsequent dynamic time warping.

    Attributes
    ----------
    matches_start_end_ : 2D array of shape (n_detected_strides x 2)
        The start (column 1) and stop (column 2) of each detected stride.
    costs_ : List of length n_detected_strides
        The cost value associated with each stride.
    acc_cost_mat_ : array with the shapes (length_template x length_data)
        The accumulated cost matrix of the DTW. The last row represents the cost function.
    cost_function_ : 1D array with the same length as the data
        The final cost function calculated as the square root of the last row of the accumulated cost matrix.
    paths_
        The full path through the cost matrix of each detected stride.

    Parameters
    ----------
    template
        The template used for matching.
        See more details in the :class:`DtwTemplates <gaitmap.stride_segmentation.dtw_templates.DtwTemplates>` docu.
        If the template has its `sampling_rate_hz` attribute set, this information is used to resample the template to
        the sampling rate of the data if `resample_template` is `True`. If `resample_template` is `False` this
        information is ignored.
    resample_template
        If `True` the template will be resampled to match the sampling rate of the data.
        This requires a valid value for `template.sampling_rate_hz`.
    max_cost
        The maximal allowed cost to find potential stride candidates in the cost function.
        Its usage depends on the exact `find_matches_method` used.
        Refer to the `find_matches_method` to learn more about this.
    min_stride_time_s
        The minimal length of a sequence in seconds to be still considered a stride.
        Matches that result in shorter sequences, will be ignored.
        At the moment this is only used if "find_peaks" is selected as `find_matches_method`.
    find_matches_method
        Select the method used to find stride candidates in the cost function.

        - "min_under_thres"
            Matches the original implementation in the paper [1].
            In this case :py:func:`.find_matches_original` will be used as method.
        - "find_peaks"
            Uses :func:`scipy.signal.find_peaks` with additional constraints to find stride candidates.
            In this case :py:func:`.find_matches_find_peaks` will be used as method.

    Other Parameters
    ----------------
    data
        The data passed to the py:meth:`segment` method.
    sampling_rate_hz
        The sampling rate of the data

    """

    template: Optional[DtwTemplate]
    max_cost: Optional[float]
    resample_template: bool
    min_stride_time_s: float
    find_matches_method: Literal["min_under_thres", "find_peaks"]

    acc_cost_mat_: np.ndarray
    paths_: Sequence[Sequence[tuple]]
    costs_: Sequence[float]

    data: Dataset
    sampling_rate_hz: float

    _allowed_methods_map = {"original": find_matches_min_under_threshold, "find_peaks": find_matches_find_peaks}

    @property
    def matches_start_end_(self) -> np.ndarray:
        """Return start and end of match."""
        return np.array([[p[0][-1], p[-1][-1]] for p in self.paths_])

    @property
    def cost_function_(self):
        """Cost function extracted from the accumulated cost matrix."""
        return np.sqrt(self.acc_cost_mat_[-1, :])

    def __init__(
        self,
        template: Optional[DtwTemplate] = None,
        resample_template: bool = True,
        find_matches_method: Literal["original", "find_peaks"] = "original",
        max_cost: Optional[float] = None,
        min_stride_time_s: Optional[float] = 0.6,
    ):
        self.template = template
        self.max_cost = max_cost
        self.min_stride_time_s = min_stride_time_s
        self.resample_template = resample_template
        self.find_matches_method = find_matches_method

    def segment(self: BaseType, data: Dataset, sampling_rate_hz: float, **_) -> BaseType:
        """Find stride candidates matching the provided template in the data.

        Parameters
        ----------
        data : array (n x m), single sensor dataframe or multi sensor dataframe
            The data array.
            n needs to be larger than `n_template`.
            m needs to be larger than `m_template`.
            Only the `m_template` first columns will be used in the matching process.
            For example if the template has 2 dimensions only `data[:, :2]` will be used.
        sampling_rate_hz
            The sampling rate of the data signal. This will be used to convert all parameters provided in seconds into
            a number of samples and it will be used to resample the template if `resample_template` is `True`.

        Returns
        -------
            self
                The class instance with all result attributes populated

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        # Validate and transform inputs
        if self.template is None:
            raise ValueError("A `template` must be specified.")

        if self.resample_template and not self.template.sampling_rate_hz:
            raise ValueError(
                "To resample the template (`resample_template=True`), a `sampling_rate_hz` must be specified for the "
                "template."
            )

        # Extract the parts of the data that is relevant for matching.
        template_array, matching_data = self._extract_relevant_data_and_template(self.template.template, self.data)

        if self.resample_template is True and sampling_rate_hz != self.template.sampling_rate_hz:
            template = self._interpolate_template(template_array, self.template.sampling_rate_hz, sampling_rate_hz)
        else:
            template = template_array

        min_distance = None
        if self.min_stride_time_s not in (None, 0, 0.0):
            min_distance = self.min_stride_time_s * sampling_rate_hz
        find_matches_method = self._allowed_methods_map.get(self.find_matches_method, None)
        if not find_matches_method:
            raise ValueError(
                'Invalid value for "find_matches_method". Must be one of {}'.format(
                    list(self._allowed_methods_map.keys())
                )
            )

        # Calculate cost matrix
        self.acc_cost_mat_ = subsequence_cost_matrix(to_time_series(template), to_time_series(matching_data))

        matches = find_matches_method(
            acc_cost_mat=self.acc_cost_mat_, max_cost=self.max_cost, min_distance=min_distance
        )
        self.paths_ = self._find_multiple_paths(self.acc_cost_mat_, matches)
        self.costs_ = np.sqrt(self.acc_cost_mat_[-1, :][matches])

        return self

    @staticmethod
    def _interpolate_template(
        template_array: np.ndarray, template_sampling_rate_hz: float, new_sampling_rate: float
    ) -> np.ndarray:
        template = resample(
            template_array, int(template_array.shape[0] * new_sampling_rate / template_sampling_rate_hz),
        )
        return template

    @staticmethod
    def _extract_relevant_data_and_template(template, data) -> Tuple[np.ndarray, np.ndarray]:
        """Get the relevant parts of the data based on the provided template and return template and data as array."""
        data_is_numpy = isinstance(data, np.ndarray)
        template_is_numpy = isinstance(template, np.ndarray)
        if data_is_numpy and template_is_numpy:
            data = np.squeeze(data)
            template = np.squeeze(template)
            if template.ndim == 1 and data.ndim == 1:
                return template, data
            if template.shape[1] > data.shape[1]:
                raise ValueError(
                    "The provided data has less columns than the used template. ({} < {})".format(
                        data.shape[1], template.shape[1]
                    )
                )
            return (
                template,
                data[:, : template.shape[1]],
            )
        data_is_df = isinstance(data, pd.DataFrame)
        template_is_df = isinstance(template, pd.DataFrame)

        if data_is_df and template_is_df:
            try:
                data = data[template.columns]
            except KeyError:
                raise KeyError(
                    "Some columns of the template are not available in the data! Missing columns: {}".format(
                        list(set(template.columns) - set(data.columns))
                    )
                )
            return template.to_numpy(), data.to_numpy()
        # TODO: Better error message
        raise ValueError("Invalid combination of data and template")

    @staticmethod
    def _find_multiple_paths(acc_cost_mat: np.ndarray, start_points: np.ndarray) -> List[np.ndarray]:
        paths = []
        for start in start_points:
            path = subsequence_path(acc_cost_mat, start)
            path_array = np.array(path)
            paths.append(path_array)
        return paths
