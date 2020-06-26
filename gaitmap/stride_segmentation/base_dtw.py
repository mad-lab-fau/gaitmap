"""A implementation of a sDTW that can be used independent of the context of Stride Segmentation."""
import warnings
from typing import Optional, Sequence, List, Tuple, Union, Dict

import numpy as np
import pandas as pd
from numba import njit
from scipy.interpolate import interp1d
from tslearn.metrics import subsequence_path, _subsequence_cost_matrix
from tslearn.utils import to_time_series
from typing_extensions import Literal

from gaitmap.base import BaseStrideSegmentation, BaseType
from gaitmap.stride_segmentation.dtw_templates import DtwTemplate
from gaitmap.utils.array_handling import find_local_minima_below_threshold, find_local_minima_with_distance
from gaitmap.utils.consts import ROI_ID_COLS
from gaitmap.utils.dataset_helper import (
    Dataset,
    is_single_sensor_dataset,
    is_multi_sensor_dataset,
    get_multi_sensor_dataset_names,
    RegionsOfInterestList,
    is_single_sensor_regions_of_interest_list,
    SingleSensorRegionsOfInterestList,
    _get_regions_of_interest_types,
    set_correct_index,
)


def find_matches_find_peaks(acc_cost_mat: np.ndarray, max_cost: float, min_distance: float) -> np.ndarray:
    """Find matches in the accumulated cost matrix using :func:`~scipy.signal.find_peaks`.

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
        More details at :func:`~gaitmap.stride_segmentation.utils.find_local_minima_below_threshold`.

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

    This uses the DTW implementation of :func:`tslearn <tslearn.metrics.subsequence_cost_matrix>`.
    This class offers a convenient wrapper around this by providing support for various datatypes to be used as inputs.
    These cover three main usecases (for examples of all of these, see the **Examples** section):

    A general purpose msDTW
        If you require a msDTW with a class based interface independent of the context of stride segmentation (or even
        this library) you can create a simple template from a (n x m) numpy array using
        :func:`~gaitmap.stride_segmentation.create_dtw_template`.
        This allows you to use just a simple numpy array as data input to the :meth:`segment` method.
        The data must have at least n samples and m columns.
        If it has more than m columns, the additional columns are ignored.
    A simple way to segment multiple sensor with the same template
        If you are using the basic datatypes of this library you can use this DTW implementation to easily apply a
        template to selected columns of multiple sensors.
        For this, the template is expected to be based on an `pd.DataFrame` wrapped in a
        :class:`~gaitmap.stride_segmentation.DtwTemplate`.
        The column names of this dataframe need to match the column names of the data the template should be applied to.
        The data can be passed as single-sensor dataframe (the columns correspond to individual sensor axis) or a
        multi-sensor dataset (either a dictionary of single-sensor dataframes or a dataframe with 2 level of column
        labels, were the upper corresponds to the sensor name.
        In case of the single-sensor-dataframe the matching and the output is identical to passing just numpy arrays.
        In case of a multi-sensor input, the provided template is applied to each of the sensors individually.
        All outputs are then dictionaries of the single-sensor outputs with the sensor name as key.
        In both cases, if a dataset has columns that are not listed in the template, they are simply ignored.
    A way to apply specific templates to specific columns
        In some cases different templates are required for different sensors.
        To do this, the template must be a dictionary of
        :class:`~gaitmap.stride_segmentation.DtwTemplate` instances, were the key
        corresponds to the sensor the template should be applied to.
        Note, that only dataframe templates are supported and **not** simple numpy array templates.
        The data input needs to be a multi-sensor dataset (see above for more information).
        The templates are then applied only to data belonging to the sensor with the same name.
        All sensors in the template dictionary must also be in the dataset.
        However, the dataset can have additional sensors, which are simply ignored.
        In this use case, the outputs are always dictionaries with the sensor name as key.

    To better understand the different datatypes have a look at the :ref:`coordinate system guide <coordinate_systems>`.

    Parameters
    ----------
    template
        The template used for matching.
        The required data type and shape depends on the use case.
        For more details view the class docstring.
        Note that the `scale` parameter of the template is used to downscale the data before the matching is performed.
        If you have data in another data range (e.g. a different unit), the scale parameter of the template should be
        adjusted.
    resample_template
        If `True` the template will be resampled to match the sampling rate of the data.
        This requires a valid value for `template.sampling_rate_hz` value.
        The resampling is performed using linear interpolation.
        Note, that this might lead to unexpected results in case of short template arrays.
    max_cost
        The maximal allowed cost to find potential match in the cost function.
        Note that the cost is roughly calculated as: `sqrt(|template - data/template.scaling|)`.
        Its usage depends on the exact `find_matches_method` used.
        Refer to the specific funtion to learn more about this.
    min_match_length_s
        The minimal length of a sequence in seconds to be considered a match.
        Matches that result in shorter sequences, will be ignored.
        In general, this exclusion is performed as a post-processing step after the matching.
        If "find_peaks" is selected as `find_matches_method`, the parameter is additionally used in the detection of
        matches directly.
    max_match_length_s
        The maximal length of a sequence in seconds to be considered a match.
        Matches that result in longer sequences will be ignored.
        This exclusion is performed as a post-processing step after the matching.
    find_matches_method
        Select the method used to find matches in the cost function.

        - "min_under_thres"
            Matches the implementation used in the paper [1]_ to detect strides in foot mounted IMUs.
            In this case :func:`~gaitmap.stride_segmentation.base_dtw.find_matches_find_peaks` will be used as method.
        - "find_peaks"
            Uses :func:`~scipy.signal.find_peaks` with additional constraints to find stride candidates.
            In this case :func:`~gaitmap.stride_segmentation.base_dtw.find_matches_min_under_threshold` will be used as
            method.

    Attributes
    ----------
    matches_start_end_ : 2D array of shape (n_detected_strides x 2) or dictionary with such values
        The start (column 1) and end (column 2) of each detected match.
    costs_ : List of length n_detected_strides or dictionary with such values
        The cost value associated with each stride.
    acc_cost_mat_ : array with the shapes (length_template x length_data) or dictionary with such values
        The accumulated cost matrix of the DTW. The last row represents the cost function.
    cost_function_ : 1D array with the same length as the data or dictionary with such values
        The final cost function calculated as the square root of the last row of the accumulated cost matrix.
    paths_ : list of arrays with length n_detected_strides or dictionary with such values
        The full path through the cost matrix of each detected stride.
        Note that the start and end values of the path might not match the start and the end values in
        `matches_start_end_`, if certain post processing steps are applied.
    matches_start_end_original_ : 2D array of shape (n_detected_strides x 2) or dictionary with such values
        Identical to `matches_start_end_` if no postprocessing is applied to change the values of start and the end of
        the matches.
        This base implementation of the DTW does not do this, but potential subclasses might modify the matches list
        during postprocessing.
        This does **not** preserve matches that were removed during postprocessing.
    roi_ids_ : List of length n_detected_strides or dictionary with such values
        The id of the region of interest each match belongs to.
        If not region of interest was specified or no matches found, this will be an empty list / a dict of empty lists.

    Other Parameters
    ----------------
    data
        The data passed to the `segment` method.
    sampling_rate_hz
        The sampling rate of the data
    regions_of_interest
        Specific regions that should be considered during matching.
        The rest of the signal is simply ignored.

    Notes
    -----
    msDTW simply calculates the DTW distance of a template at every possible timepoint in the signal.
    While the template is warped, it is advisable to use a template that has a similar length than the expected matches.
    Using `resample_template` can help with that.
    Further, the template should cover the same signal range than the original signal.
    You can use the `scale` parameter of the :class:`~gaitmap.stride_segmentation.DtwTemplate` to adapt your template
    to your data.

    If you see unexpected matches or missing matches in your results, it is advisable to plot `acc_cost_mat_` and
    `cost_function_`.
    They can provide insight in the matching process.

    .. [1] Barth, J., Oberndorfer, C., Kugler, P., Schuldhaus, D., Winkler, J., Klucken, J., & Eskofier, B. (2013).
       Subsequence dynamic time warping as a method for robust step segmentation using gyroscope signals of daily life
       activities. Proceedings of the Annual International Conference of the IEEE Engineering in Medicine and Biology
       Society, EMBS, 6744–6747. https://doi.org/10.1109/EMBC.2013.6611104

    Examples
    --------
    Running a simple matching using arrays as input:

    >>> from gaitmap.stride_segmentation import create_dtw_template
    >>> template_data = np.array([1, 2, 1])
    >>> data = np.array([0, 0, 1, 2, 1, 0, 1, 2, 1, 0])
    >>> template = create_dtw_template(template_data)
    >>> dtw = BaseDtw(template=template, max_cost=1, resample_template=False)
    >>> dtw = dtw.segment(data, sampling_rate_hz=1)  # Sampling rate is not important for this example
    >>> dtw.matches_start_end_
    array([[2, 4],
           [6, 8]])

    """

    template: Optional[DtwTemplate]
    max_cost: Optional[float]
    resample_template: bool
    min_match_length_s: Optional[float]
    max_match_length_s: Optional[float]
    find_matches_method: Literal["min_under_thres", "find_peaks"]

    matches_start_end_: Union[np.ndarray, Dict[str, np.ndarray]]
    acc_cost_mat_: Union[np.ndarray, Dict[str, np.ndarray]]
    paths_: Union[Sequence[Sequence[tuple]], Dict[str, Sequence[Sequence[tuple]]]]
    costs_: Union[Sequence[float], Dict[str, Sequence[float]]]
    roi_ids_: Union[Sequence[float], Dict[str, Sequence[float]]]

    data: Union[np.ndarray, Dataset]
    regions_of_interest: Optional[RegionsOfInterestList]
    sampling_rate_hz: float

    _allowed_methods_map = {"min_under_thres": find_matches_min_under_threshold, "find_peaks": find_matches_find_peaks}
    _min_sequence_length: Optional[float]
    _max_sequence_length: Optional[float]
    _roi_type = Union[Optional[str], Dict[str, Optional[str]]]

    @property
    def cost_function_(self):
        """Cost function extracted from the accumulated cost matrix."""
        if isinstance(self.acc_cost_mat_, dict):
            return {s: np.sqrt(cost_mat[-1, :]) for s, cost_mat in self.acc_cost_mat_.items()}
        return np.sqrt(self.acc_cost_mat_[-1, :])

    @property
    def matches_start_end_original_(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Return the starts and end directly from the paths.

        This will not be effected by potential changes of the postprocessing.
        """
        if isinstance(self.acc_cost_mat_, dict):
            return {s: np.array([[p[0][-1], p[-1][-1]] for p in path]) for s, path in self.paths_.items()}
        return np.array([[p[0][-1], p[-1][-1]] for p in self.paths_])

    def __init__(
        self,
        template: Optional[Union[DtwTemplate, Dict[str, DtwTemplate]]] = None,
        resample_template: bool = True,
        find_matches_method: Literal["min_under_thres", "find_peaks"] = "find_peaks",
        max_cost: Optional[float] = None,
        min_match_length_s: Optional[float] = None,
        max_match_length_s: Optional[float] = None,
    ):
        self.template = template
        self.max_cost = max_cost
        self.min_match_length_s = min_match_length_s
        self.max_match_length_s = max_match_length_s
        self.resample_template = resample_template
        self.find_matches_method = find_matches_method

    def segment(
        self: BaseType,
        data: Union[np.ndarray, Dataset],
        sampling_rate_hz: float,
        regions_of_interest: Optional[RegionsOfInterestList] = None,
        **kwargs,
    ) -> BaseType:
        """Find matches by warping the provided template to the data.

        Parameters
        ----------
        data : array, single-sensor dataframe, or multi-sensor dataset
            The input data.
            For details on the required datatypes review the class docstring.
        regions_of_interest
            Regions of interest in the signal that should be considered for matching.
            All signal outside these regions is ignored.
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
        self.regions_of_interest = regions_of_interest

        # TODO: Check region of interest Dtype

        # Validate and transform inputs
        if self.template is None:
            raise ValueError("A `template` must be specified.")

        if self.find_matches_method not in self._allowed_methods_map:
            raise ValueError(
                'Invalid value for "find_matches_method". Must be one of {}'.format(
                    list(self._allowed_methods_map.keys())
                )
            )

        template = self.template
        if isinstance(data, np.ndarray) or is_single_sensor_dataset(data, check_gyr=False, check_acc=False):
            # Single template single sensor: easy
            (
                self.acc_cost_mat_,
                self.paths_,
                self.costs_,
                self.matches_start_end_,
                self.roi_ids_,
                self._roi_type,
            ) = self._segment_single_dataset(data, template, regions_of_interest)
        elif is_multi_sensor_dataset(data, check_gyr=False, check_acc=False):
            if isinstance(template, dict):
                # multiple templates, multiple sensors: Apply the correct template to the correct sensor.
                # Ignore the rest
                results = dict()
                for sensor, single_template in template.items():
                    roi = self._get_region_of_interest_for_sensor(sensor)
                    results[sensor] = self._segment_single_dataset(data[sensor], single_template, roi)
            elif is_single_sensor_dataset(template.data, check_gyr=False, check_acc=False):
                # single template, multiple sensors: Apply template to all sensors
                results = dict()
                for sensor in get_multi_sensor_dataset_names(data):
                    roi = self._get_region_of_interest_for_sensor(sensor)
                    results[sensor] = self._segment_single_dataset(data[sensor], template, roi)
            else:
                raise ValueError(
                    "In case of a multi-sensor dataset input, the used template must either be of type "
                    "`Dict[str, DtwTemplate]` or the template array must have the shape of a single-sensor dataframe."
                )
            self.acc_cost_mat_, self.paths_, self.costs_, self.matches_start_end_, self.roi_ids_, self._roi_type = (
                dict(),
                dict(),
                dict(),
                dict(),
                dict(),
                dict(),
            )
            for sensor, r in results.items():
                self.acc_cost_mat_[sensor] = r[0]
                self.paths_[sensor] = r[1]
                self.costs_[sensor] = r[2]
                self.matches_start_end_[sensor] = r[3]
                self.roi_ids_[sensor] = r[4]
                self._roi_type = r[5]
        else:
            # TODO: Better error message -> This will be fixed globally
            raise ValueError("The type or shape of the provided dataset is not supported.")
        return self

    def _segment_single_dataset(self, dataset, template, roi: Optional[SingleSensorRegionsOfInterestList]):
        if self.resample_template and not template.sampling_rate_hz:
            raise ValueError(
                "To resample the template (`resample_template=True`), a `sampling_rate_hz` must be specified for the "
                "template."
            )
        if (
            template.sampling_rate_hz
            and self.sampling_rate_hz != template.sampling_rate_hz
            and self.resample_template is False
        ):
            warnings.warn(
                "The data and template sampling rate are different ({} Hz vs. {} Hz), "
                "but `resample_template` is False. "
                "This might lead to unexpected results".format(template.sampling_rate_hz, self.sampling_rate_hz)
            )

        # Extract the parts of the data that is relevant for matching.
        template_array, matching_data = self._extract_relevant_data_and_template(template.data, dataset)
        # Ensure that all values are floats
        template_array = template_array.astype(float)
        matching_data = matching_data.astype(float)
        # Downscale the data by the factor provided by the template
        scaling_factor = float(getattr(template, "scaling", None) or 1.0)
        matching_data /= scaling_factor

        if self.resample_template is True and self.sampling_rate_hz != template.sampling_rate_hz:
            final_template = self._resample_template(template_array, template.sampling_rate_hz, self.sampling_rate_hz)
        else:
            final_template = template_array

        self._min_sequence_length = self.min_match_length_s
        if self._min_sequence_length not in (None, 0, 0.0):
            self._min_sequence_length *= self.sampling_rate_hz
        self._max_sequence_length = self.max_match_length_s
        if self._max_sequence_length not in (None, 0, 0.0):
            self._max_sequence_length *= self.sampling_rate_hz

        find_matches_method = self._allowed_methods_map[self.find_matches_method]

        if roi is not None:
            roi_start_end = roi[["start", "end"]].to_numpy()
            roi_type = _get_regions_of_interest_types(roi.reset_index().columns)
            roi = set_correct_index(roi, [ROI_ID_COLS[roi_type]])
        else:
            roi_start_end = np.array([[0, len(matching_data)]])
            roi_type = None

        # Calculate cost matrix
        acc_cost_mat_ = self._calculate_cost_matrix(final_template, matching_data, roi_start_end)
        # TODO: Rework the ROI concept here and how to report the costmat/costfunc in case of ROI
        matches = self._find_matches(
            acc_cost_mat=acc_cost_mat_,
            max_cost=self.max_cost,
            min_sequence_length=self._min_sequence_length,
            find_matches_method=find_matches_method,
        )
        if len(matches) == 0:
            paths_ = []
            costs_ = []
            matches_start_end_ = []
            roi_id_ = []
        else:
            paths_ = self._find_multiple_paths(acc_cost_mat_, np.sort(matches))
            matches_start_end_ = np.array([[p[0][-1], p[-1][-1]] for p in paths_])
            # Calculate cost before potential modifications are made to start and end
            costs_ = np.sqrt(acc_cost_mat_[-1, :][matches_start_end_[:, 1]])
            to_keep = np.ones(len(matches_start_end_)).astype(bool)
            matches_start_end_, to_keep = self._postprocess_matches(
                data=dataset, paths=paths_, cost=costs_, matches_start_end=matches_start_end_, to_keep=to_keep
            )
            matches_start_end_ = matches_start_end_[to_keep]
            self._post_postprocess_check(matches_start_end_)
            paths_ = [p for i, p in enumerate(paths_) if i in np.where(to_keep)[0]]
            if roi is None:
                roi_id_ = []
            else:
                roi_id_ = np.ones(len(matches_start_end_))
                for i, (start, end) in roi[["start", "end"]].iterrows():
                    roi_id_[np.where((matches_start_end_[0] > start) & (matches_start_end_[1] < end))[0]] = i
        return acc_cost_mat_, paths_, costs_, matches_start_end_, roi_id_, roi_type

    def _calculate_cost_matrix(self, template, matching_data, rois_start_end):  # noqa: no-self-use
        template = to_time_series(template)
        matching_data = to_time_series(matching_data)
        return _multi_roi_dtw_cost_mat(template, matching_data, rois_start_end)

    def _find_matches(self, acc_cost_mat, max_cost, min_sequence_length, find_matches_method):  # noqa: no-self-use
        return find_matches_method(acc_cost_mat=acc_cost_mat, max_cost=max_cost, min_distance=min_sequence_length)

    def _postprocess_matches(
        self,
        data,  # noqa: unused-argument
        paths: List,  # noqa: unused-argument
        cost: np.ndarray,  # noqa: unused-argument
        matches_start_end: np.ndarray,
        to_keep: np.array,
    ) -> Tuple[np.ndarray, np.array]:
        """Apply postprocessing.

        This can be overwritten by subclasses to filter and modify the matches further.
        Note, that no values from matches_start_stop or paths should be deleted (only modified).
        If a stride needs to be deleted, its index from the **original** matches_start_end list should be added to
        the `to_remove` boolmap should be updated.

        Parameters
        ----------
        data
            The actual raw data
        paths
            The identified paths through the cost matrix
        cost
            The overall cost of each match
        matches_start_end
            The start and end of each match.
            This should either be modified or returned without modification.
        to_keep
            A boolmap indicating which strides should be kept.
            This should either be modified or returned without modification.

        Returns
        -------
        matches_start_end
            A modified version of the start-end array.
            No strides should be removed! Just modifications to the actual values
        to_keep
            A boolmap with the length of nstrides. It indicates if a stride should be kept (True) or removed later (
            False)

        """
        # Remove matches that are shorter that min_match_length
        min_sequence_length = self._min_sequence_length
        if min_sequence_length is not None:
            # only select the once that are not already removed
            indices = np.where(to_keep)[0]
            matches_start_end_valid = matches_start_end[indices]
            invalid_strides = (
                np.abs(matches_start_end_valid[:, 1] - matches_start_end_valid[:, 0]) <= min_sequence_length
            )
            to_keep[indices[invalid_strides]] = False
        max_sequence_length = self._max_sequence_length
        if max_sequence_length is not None:
            # only select the once that are not already removed
            indices = np.where(to_keep)[0]
            matches_start_end_valid = matches_start_end[indices]
            invalid_strides = (
                np.abs(matches_start_end_valid[:, 1] - matches_start_end_valid[:, 0]) >= max_sequence_length
            )
            to_keep[indices[invalid_strides]] = False
        return matches_start_end, to_keep

    def _post_postprocess_check(self, matches_start_end):
        """Check that is invoked after all processing is done.

        Parameters
        ----------
        matches_start_end
            The start and end of all matches remaining after the postprocessing

        """

    def _get_region_of_interest_for_sensor(self, sensor_name: str):
        if self.regions_of_interest is None or is_single_sensor_regions_of_interest_list(self.regions_of_interest):
            return self.regions_of_interest
        return self.regions_of_interest.get(sensor_name, None)

    @staticmethod
    def _resample_template(
        template_array: np.ndarray, template_sampling_rate_hz: float, new_sampling_rate: float
    ) -> np.ndarray:
        len_template = template_array.shape[0]
        current_x = np.linspace(0, len_template, len_template)
        template = interp1d(current_x, template_array, axis=0)(
            np.linspace(0, len_template, int(len_template * new_sampling_rate / template_sampling_rate_hz))
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
                    "Some columns of the template are not available in the data! This might happen because you "
                    "provided the data in the wrong coordinate frame (Sensor vs. Body)."
                    "Review the general documentation for more information."
                    "\n\nMissing columns: {}".format(list(set(template.columns) - set(data.columns)))
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


@njit(cache=True, parallel=True)
def _multi_roi_dtw_cost_mat(template: np.ndarray, matching_data: np.ndarray, rois_start_end: np.ndarray):
    l1 = template.shape[0]
    l2 = matching_data.shape[0]
    cost_matrix = np.full((l1, l2), np.inf)
    for start, end in rois_start_end:
        cost_matrix[:, start : end + 1] = _subsequence_cost_matrix(template, matching_data[start : end + 1])
    return cost_matrix
