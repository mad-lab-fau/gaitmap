"""The msDTW based stride segmentation algorithm by Barth et al 2013."""
from typing import Optional, Sequence, List

import numpy as np
from scipy.signal import resample
from tslearn.metrics import subsequence_cost_matrix, subsequence_path
from tslearn.utils import to_time_series
from typing_extensions import Literal

from gaitmap.base import BaseStrideSegmentation, BaseType
from gaitmap.stride_segmentation.utils import find_local_minima_with_distance, find_local_minima_below_threshold


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


def find_matches_original(acc_cost_mat: np.ndarray, max_cost: float, **_) -> np.ndarray:
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


class BarthDtw(BaseStrideSegmentation):
    """Segment strides using a single stride template and Dynamic Time Warping.

    BarthDtw uses a manually created template of an IMU stride to find multiple occurrences of similar signals in a
    continuous data stream.
    The method is not limited to a single sensor-axis or sensor, but can use any number of dimensions of the provided
    input signal simultaneously.
    For more details refer to the `Notes` section.

    Attributes
    ----------
    strides_start_end_ : 2D array of shape (n_detected_strides x 2)
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
    template : (n x m) array representing a single stride
        The template of length n used for matching. If the template has multiple dimensions m, the first m dimensions of
        the data are used to perform the matching.
    template_sampling_rate_hz
        Sampling rate used for the template. This information is used to resample the template to the sampling rate of
        the data if `resample_template` is `True`. If `resample_template` is `False` this information is ignored.
    resample_template
        If `True` the template will be resampled to match the sampling rate of the data.
        This requires a valid value for `template_sampling_rate_hz`.
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

        - "original"
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

    Notes
    -----
    TODO: Add additional details about the use of DTW for stride segmentation

    .. [1] Barth, J., Oberndorfer, C., Kugler, P., Schuldhaus, D., Winkler, J., Klucken, J., & Eskofier, B. (2013).
       Subsequence dynamic time warping as a method for robust step segmentation using gyroscope signals of daily life
       activities. Proceedings of the Annual International Conference of the IEEE Engineering in Medicine and Biology
       Society, EMBS, 6744–6747. https://doi.org/10.1109/EMBC.2013.6611104

    """

    template: Optional[np.ndarray]
    template_sampling_rate_hz: Optional[float]
    max_cost: Optional[float]
    resample_template: bool
    min_stride_time_s: float
    find_matches_method: Literal["original", "find_peaks"]

    acc_cost_mat_: np.ndarray
    paths_: Sequence[Sequence[tuple]]
    costs_: Sequence[float]

    data: np.ndarray
    sampling_rate_hz: float

    _allowed_methods_map = {"original": find_matches_original, "find_peaks": find_matches_find_peaks}

    @property
    def strides_start_end_(self) -> np.ndarray:
        """Return start and end of each stride candidate."""
        return np.array([[p[0][-1], p[-1][-1]] for p in self.paths_])

    @property
    def cost_function_(self):
        """Cost function extracted from the accumulated cost matrix."""
        return np.sqrt(self.acc_cost_mat_[-1, :])

    def __init__(
        self,
        template: Optional[np.ndarray] = None,
        template_sampling_rate_hz: Optional[float] = None,
        resample_template: bool = True,
        find_matches_method: Literal["original", "find_peaks"] = "original",
        max_cost: Optional[float] = None,
        min_stride_time_s: Optional[float] = 0.6,
    ):
        self.template = template
        self.template_sampling_rate_hz = template_sampling_rate_hz
        self.max_cost = max_cost
        self.min_stride_time_s = min_stride_time_s
        self.resample_template = resample_template
        self.find_matches_method = find_matches_method

    def segment(self: BaseType, data: np.ndarray, sampling_rate_hz: float, **_) -> BaseType:
        """Find stride candidates matching the provided template in the data.

        Parameters
        ----------
        data : array (n x m)
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
        # TODO: Test multidimensional matchings
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        # Validate and transform inputs
        if self.template is None:
            raise ValueError("A `template` must be specified.")

        if self.resample_template and not self.template_sampling_rate_hz:
            raise ValueError(
                "To resample the template (`resample_template=True`), `template_sampling_rate_hz` must be specified."
            )

        if self.resample_template is True and sampling_rate_hz != self.template_sampling_rate_hz:
            template = self._interpolate_template(sampling_rate_hz)
        else:
            template = self.template

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
        self.acc_cost_mat_ = subsequence_cost_matrix(to_time_series(template), to_time_series(data))

        matches = find_matches_method(
            acc_cost_mat=self.acc_cost_mat_, max_cost=self.max_cost, min_distance=min_distance
        )
        self.paths_ = self._find_multiple_paths(self.acc_cost_mat_, matches)
        self.costs_ = np.sqrt(self.acc_cost_mat_[-1, :][matches])

        return self

    def _interpolate_template(self, new_sampling_rate: float) -> np.ndarray:
        template = resample(
            self.template, int(self.template.shape[0] * new_sampling_rate / self.template_sampling_rate_hz)
        )
        return template

    @staticmethod
    def _find_multiple_paths(acc_cost_mat: np.ndarray, start_points: np.ndarray) -> List[np.ndarray]:
        paths = []
        for start in start_points:
            path = subsequence_path(acc_cost_mat, start)
            path_array = np.array(path)
            paths.append(path_array)
        return paths
