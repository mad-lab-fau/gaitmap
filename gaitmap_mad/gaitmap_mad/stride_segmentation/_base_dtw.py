"""A implementation of a sDTW that can be used independent of the context of Stride Segmentation."""
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
from joblib import Memory
from numba import njit
from scipy.interpolate import interp1d
from typing_extensions import Literal

from gaitmap.base import BaseAlgorithm
from gaitmap.utils._algo_helper import invert_result_dictionary, set_params_from_dict
from gaitmap.utils._types import _Hashable
from gaitmap.utils.array_handling import find_local_minima_below_threshold, find_local_minima_with_distance
from gaitmap.utils.datatype_helper import SensorData, get_multi_sensor_names, is_sensor_data, is_single_sensor_data
from gaitmap_mad.stride_segmentation._dtw_templates import BaseDtwTemplate, DtwTemplate
from gaitmap_mad.stride_segmentation._vendored_tslearn import (
    _local_squared_dist,
    subsequence_cost_matrix,
    subsequence_path,
)

Self = TypeVar("Self", bound="BaseDtw")


def _timeseries_reshape(data: np.ndarray) -> np.ndarray:
    """Add a dummy dimension to the time series."""
    if data.ndim <= 1:
        data = data.reshape((-1, 1))
    return data


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
    gaitmap.utils.array_handling.find_local_minima_with_distance: Details on the function call to `find_peaks`.
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
        More details at :func:`~gaitmap.utils.array_handling.find_local_minima_below_threshold`.

    Returns
    -------
    list_of_matches
        A list of indices marking the end of a potential stride.

    See Also
    --------
    gaitmap.utils.array_handling.find_local_minima_below_threshold: Implementation details.

    """
    return find_local_minima_below_threshold(np.sqrt(acc_cost_mat[-1, :]), threshold=max_cost)


class BaseDtw(BaseAlgorithm):
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
            In this case :func:`~gaitmap.stride_segmentation.base_dtw.find_matches_min_under_threshold` will be used as
            method.
        - "find_peaks"
            Uses :func:`~scipy.signal.find_peaks` with additional constraints to find stride candidates.
            In this case :func:`~gaitmap.stride_segmentation.base_dtw.find_matches_find_peaks` will be used as
            method.
    memory
        An optional `joblib.Memory` object that can be provided to cache the creation of cost matrizes and the peak
        detection.

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

    Other Parameters
    ----------------
    data
        The data passed to the `segment` method.
    sampling_rate_hz
        The sampling rate of the data

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

    >>> from gaitmap.stride_segmentation import DtwTemplate
    >>> template_data = np.array([1, 2, 1])
    >>> data = np.array([0, 0, 1, 2, 1, 0, 1, 2, 1, 0])
    >>> template = DtwTemplate(template_data)
    >>> dtw = BaseDtw(template=template, max_cost=1, resample_template=False)
    >>> dtw = dtw.segment(data, sampling_rate_hz=1)  # Sampling rate is not important for this example
    >>> dtw.matches_start_end_
    array([[2, 4],
           [6, 8]])

    """

    _action_methods = ("segment",)

    template: Optional[Union[BaseDtwTemplate, Dict[_Hashable, BaseDtwTemplate]]]
    max_cost: Optional[float]
    resample_template: bool
    min_match_length_s: Optional[float]
    max_match_length_s: Optional[float]
    max_template_stretch_ms: Optional[float]
    max_signal_stretch_ms: Optional[float]
    find_matches_method: Literal["min_under_thres", "find_peaks"]
    memory: Optional[Memory]

    matches_start_end_: Union[np.ndarray, Dict[_Hashable, np.ndarray]]
    acc_cost_mat_: Union[np.ndarray, Dict[_Hashable, np.ndarray]]
    paths_: Union[List[np.ndarray], Dict[_Hashable, List[np.ndarray]]]
    costs_: Union[np.ndarray, Dict[_Hashable, np.ndarray]]

    data: Union[np.ndarray, SensorData]
    sampling_rate_hz: float

    _allowed_methods_map = {"min_under_thres": find_matches_min_under_threshold, "find_peaks": find_matches_find_peaks}
    _min_sequence_length: Optional[float]
    _max_sequence_length: Optional[float]
    _max_template_stretch: Optional[int]
    _max_signal_stretch: Optional[int]

    @property
    def cost_function_(self):
        """Cost function extracted from the accumulated cost matrix."""
        if isinstance(self.acc_cost_mat_, dict):
            return {s: np.sqrt(cost_mat[-1, :]) for s, cost_mat in self.acc_cost_mat_.items()}
        return np.sqrt(self.acc_cost_mat_[-1, :])

    @property
    def matches_start_end_original_(
        self,
    ) -> Union[np.ndarray, Dict[_Hashable, np.ndarray]]:
        """Return the starts and end directly from the paths.

        This will not be effected by potential changes of the postprocessing.
        """
        if isinstance(self.paths_, dict):
            # We add +1 here to adhere to the convention that the end index of a region/stride is exclusive.
            return {s: np.array([[p[0][-1], p[-1][-1] + 1] for p in path]) for s, path in self.paths_.items()}
        return np.array([[p[0][-1], p[-1][-1] + 1] for p in self.paths_])

    def __init__(
        self,
        template: Optional[Union[DtwTemplate, Dict[_Hashable, DtwTemplate]]] = None,
        resample_template: bool = True,
        find_matches_method: Literal["min_under_thres", "find_peaks"] = "find_peaks",
        max_cost: Optional[float] = None,
        min_match_length_s: Optional[float] = None,
        max_match_length_s: Optional[float] = None,
        max_template_stretch_ms: Optional[float] = None,
        max_signal_stretch_ms: Optional[float] = None,
        memory: Optional[Memory] = None,
    ):
        self.template = template
        self.max_cost = max_cost
        self.min_match_length_s = min_match_length_s
        self.max_match_length_s = max_match_length_s
        self.max_template_stretch_ms = max_template_stretch_ms
        self.max_signal_stretch_ms = max_signal_stretch_ms
        self.resample_template = resample_template
        self.find_matches_method = find_matches_method
        self.memory = memory

    def segment(self: Self, data: Union[np.ndarray, SensorData], sampling_rate_hz: float, **_) -> Self:
        """Find matches by warping the provided template to the data.

        Parameters
        ----------
        data : array, single-sensor dataframe, or multi-sensor dataset
            The input data.
            For details on the required datatypes review the class docstring.
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

        # We seperate calculating everything from actually setting the results, to provide a better insert point for
        # caching.
        results = self._segment(data=self.data, sampling_rate_hz=self.sampling_rate_hz, memory=self.memory)
        set_params_from_dict(self, results, result_formatting=True)
        return self

    def _segment(
        self,
        data: Union[np.ndarray, SensorData],
        sampling_rate_hz: float,
        memory: Optional[Memory] = None,
    ) -> Dict[str, Any]:
        if not memory:
            memory = Memory(None)

        self._validate_basic_inputs()

        self._min_sequence_length = self.min_match_length_s
        if self._min_sequence_length is not None:
            self._min_sequence_length *= sampling_rate_hz
        self._max_sequence_length = self.max_match_length_s
        if self._max_sequence_length is not None:
            self._max_sequence_length *= sampling_rate_hz

        # For the typechecker
        assert self.template is not None

        results: Union[
            Dict[str, Union[np.ndarray, List[np.ndarray]]],
            Dict[str, Dict[Union[_Hashable, str], Union[np.ndarray, List[np.ndarray]]]],
        ]
        if isinstance(data, np.ndarray):
            dataset_type = "array"
        else:
            dataset_type = is_sensor_data(data, check_gyr=False, check_acc=False)
        template = self.template
        if dataset_type in ("single", "array"):
            # Single template single sensor: easy
            results = self._segment_single_dataset(data, template, memory=memory)
        else:  # Multisensor
            result_dict: Dict[_Hashable, Dict[str, Union[np.ndarray, List[np.ndarray]]]] = {}
            if isinstance(template, dict):
                # multiple templates, multiple sensors: Apply the correct template to the correct sensor.
                # Ignore the rest
                for sensor, single_template in template.items():
                    result_dict[sensor] = self._segment_single_dataset(data[sensor], single_template, memory=memory)
            elif is_single_sensor_data(template.get_data(), check_gyr=False, check_acc=False):
                # single template, multiple sensors: Apply template to all sensors
                for sensor in get_multi_sensor_names(data):
                    result_dict[sensor] = self._segment_single_dataset(data[sensor], template, memory=memory)
            else:
                raise ValueError(
                    "In case of a multi-sensor dataset input, the used template must either be of type "
                    "`Dict[str, DtwTemplate]` or the template array must have the shape of a single-sensor dataframe."
                )
            results = invert_result_dictionary(result_dict)

        return results

    def _segment_single_dataset(
        self, dataset, template, memory: Memory
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
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

        # Extract the parts of the data that is relevant for matching and apply potential data transforms defined in
        # the template.
        template_array, matching_data = self._extract_relevant_data_and_template(
            template, dataset, self.sampling_rate_hz
        )
        # Ensure that all values are floats
        template_array = template_array.astype(float)
        matching_data = matching_data.astype(float)
        if self.resample_template is True and self.sampling_rate_hz != template.sampling_rate_hz:
            final_template = self._resample_template(template_array, template.sampling_rate_hz, self.sampling_rate_hz)
        else:
            final_template = template_array

        max_template_stretch, max_signal_stretch = self._calculate_constrains(template)

        find_matches_method = self._allowed_methods_map[self.find_matches_method]
        cost_matrix_method, cost_matrix_kwargs = self._select_cost_matrix_method(
            max_template_stretch, max_signal_stretch
        )

        # If we have smart cache enabled, this will cache the methods with the longest runtime
        find_matches_method = memory.cache(find_matches_method)
        cost_matrix_method = memory.cache(cost_matrix_method)

        # Calculate cost matrix
        # We need to copy the result to ensure that it is an actual array and not a view on an array.
        acc_cost_mat_ = cost_matrix_method(
            _timeseries_reshape(final_template), _timeseries_reshape(matching_data), **cost_matrix_kwargs
        ).copy()

        # Find matches and postprocess them
        matches = self._find_matches(
            acc_cost_mat=acc_cost_mat_,
            max_cost=self.max_cost,
            min_sequence_length=self._min_sequence_length,
            find_matches_method=find_matches_method,
        )
        if len(matches) == 0:
            paths_ = []
            costs_ = np.array([])
            matches_start_end_ = np.array([])
        else:
            paths_ = self._find_multiple_paths(acc_cost_mat_, np.sort(matches))
            # We add +1 here to adhere to the convention that the end index of a region/stride is exclusive.
            matches_start_end_ = np.array([[p[0][-1], p[-1][-1] + 1] for p in paths_])
            # Calculate cost before potential modifications are made to start and end
            costs_ = np.sqrt(acc_cost_mat_[-1, :][matches_start_end_[:, 1] - 1])
            to_keep = np.ones(len(matches_start_end_)).astype(bool)
            matches_start_end_, to_keep = self._postprocess_matches(
                data=dataset,
                paths=paths_,
                cost=costs_,
                matches_start_end=matches_start_end_,
                to_keep=to_keep,
                acc_cost_mat=acc_cost_mat_,
                memory=memory,
            )
            matches_start_end_ = matches_start_end_[to_keep]
            self._post_postprocess_check(matches_start_end_)
            paths_ = [p for i, p in enumerate(paths_) if i in np.where(to_keep)[0]]
        return {
            "acc_cost_mat": acc_cost_mat_,
            "paths": paths_,
            "costs": costs_,
            "matches_start_end": matches_start_end_,
        }

    def _select_cost_matrix_method(  # noqa: no-self-use
        self, max_template_stretch: float, max_signal_stretch: float
    ) -> Tuple[Callable, Dict[str, Any]]:
        """Select the correct function to calculate the cost matrix.

        This is separate method to make it easy to overwrite by a subclass.
        """
        # In case we don't have local constrains, we can use the simple dtw implementation
        if max_template_stretch == np.inf and max_signal_stretch == np.inf:
            return subsequence_cost_matrix, {}
        # ... otherwise, we use our custom cost matrix. This is slower. Therefore, we only want to use when we need it.
        return (
            subsequence_cost_matrix_with_constrains,
            {
                "max_subseq_steps": max_template_stretch,
                "max_longseq_steps": max_signal_stretch,
                "return_only_cost": True,
            },
        )

    def _find_matches(self, acc_cost_mat, max_cost, min_sequence_length, find_matches_method):  # noqa: no-self-use
        """Find the matches in the cost matrix.

        This is separate method to make it easy to overwrite by a subclass.
        """
        return find_matches_method(
            acc_cost_mat=acc_cost_mat,
            max_cost=max_cost,
            min_distance=min_sequence_length,
        )

    def _postprocess_matches(
        self,
        data,  # noqa: unused-argument
        paths: List,  # noqa: unused-argument
        cost: np.ndarray,  # noqa: unused-argument
        matches_start_end: np.ndarray,
        acc_cost_mat: np.ndarray,  # noqa: unused-argument
        to_keep: np.ndarray,
        memory: Memory,  # noqa: unused-argument
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        acc_cost_mat
            The accumulated cost matrix of the DTW algorithm.
        memory
            A joblib memory instance that can be used to cache slow parts of the calculation.
            Note, that actually caching will only be performed, if smart caching is enabled.

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

    @staticmethod
    def _resample_template(
        template_array: np.ndarray,
        template_sampling_rate_hz: float,
        new_sampling_rate: float,
    ) -> np.ndarray:
        len_template = template_array.shape[0]
        current_x = np.linspace(0, len_template, len_template)
        template = interp1d(current_x, template_array, axis=0)(
            np.linspace(
                0,
                len_template,
                int(len_template * new_sampling_rate / template_sampling_rate_hz),
            )
        )
        return template

    @staticmethod
    def _extract_relevant_data_and_template(
        template: DtwTemplate,
        data: Union[np.ndarray, pd.DataFrame],
        sampling_rate_hz: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the relevant parts of the data based on the provided template and return template and data as array."""
        template_array = template.get_data()
        data_is_numpy = isinstance(data, np.ndarray)
        template_is_numpy = isinstance(template_array, np.ndarray)
        if data_is_numpy and template_is_numpy:
            data = np.squeeze(data)
            template_array = np.squeeze(template_array)
            if template_array.ndim == 1 and data.ndim == 1:
                return template_array, data
            if template_array.shape[1] > data.shape[1]:
                raise ValueError(
                    "The provided data has less columns than the used template. ({} < {})".format(
                        data.shape[1], template_array.shape[1]
                    )
                )
            return (
                template_array,
                data[:, : template_array.shape[1]],
            )
        data_is_df = isinstance(data, pd.DataFrame)
        template_is_df = isinstance(template_array, pd.DataFrame)

        if data_is_df and template_is_df:
            try:
                data = data[template_array.columns]
                # We transform the data here based on the scaling/preprocessing defined in the template
                data = template.transform_data(data, sampling_rate_hz)
            except KeyError as e:
                raise KeyError(
                    "Some columns of the template are not available in the data! This might happen because you "
                    "provided the data in the wrong coordinate frame (Sensor vs. Body)."
                    "Review the general documentation for more information."
                    "\n\nMissing columns: {}".format(list(set(template_array.columns) - set(data.columns)))
                ) from e
            return template_array.to_numpy(), data.to_numpy()
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

    def _validate_basic_inputs(self):
        if self.template is None:
            raise ValueError("A `template` must be specified.")

        if self.find_matches_method not in self._allowed_methods_map:
            raise ValueError(
                "Invalid value for `find_matches_method`. Must be one of {}".format(
                    list(self._allowed_methods_map.keys())
                )
            )

        if self.max_template_stretch_ms is not None and self.max_template_stretch_ms <= 0:
            raise ValueError(
                "Invalid value for `max_template_stretch_ms`."
                "The value must be a number larger than 0 and not {}".format(self.max_template_stretch_ms)
            )
        if self.max_signal_stretch_ms is not None and self.max_signal_stretch_ms <= 0:
            raise ValueError(
                "Invalid value for `max_signal_stretch_ms`."
                "The value must be a number larger than 0 and not {}".format(self.max_signal_stretch_ms)
            )

    def _calculate_constrains(self, template):
        _max_template_stretch = self.max_template_stretch_ms
        _max_signal_stretch = self.max_signal_stretch_ms
        if _max_signal_stretch and template.sampling_rate_hz is None:
            raise ValueError(
                "To use the local warping constraint for the template, a `sampling_rate_hz` must be specified for "
                "the template."
            )

        if _max_template_stretch is None:
            _max_template_stretch = np.inf
        else:
            # Use the correct template sampling rate
            sampling_rate = self.sampling_rate_hz
            if self.resample_template is False:
                sampling_rate = template.sampling_rate_hz
            _max_template_stretch = np.round(_max_template_stretch / 1000 * sampling_rate)
        if _max_signal_stretch is None:
            _max_signal_stretch = np.inf
        else:
            _max_signal_stretch = np.round(_max_signal_stretch / 1000 * self.sampling_rate_hz)
        return _max_template_stretch, _max_signal_stretch


def subsequence_cost_matrix_with_constrains(
    subseq, longseq, max_subseq_steps, max_longseq_steps, return_only_cost=False
):
    """Create a costmatrix using local warping constrains.

    This works, by tracking for each step, how many consecutive vertical (subseq) and how many horizontal (longseq)
    steps have been taken to reach a certain point in the cost matrix.
    Whenever, a step in another direction is taken, all other counters are reset to 0.
    If more or equal than `max_subseq_steps` consecutive vertical or `max_longseq_steps` consecutive horizontal steps
    have been taken, it is not possible to take another vertical or horizontal, respectively, unless at least one
    step in any other direction is taken.

    To save ram, the implementation, only tracks a single counter.
    Negative values of the counter indicate that we are currently in a series of horizontal steps (along the `longseq`).
    Positive values indicate that we are in a series of vertical steps.
    In result, the returned cost-matrix has 2 "layers".
    The first layer is the actual warping cost.
    The second layer are the respective counters.
    """
    cum_sum = _subsequence_cost_matrix_with_constrains(subseq, longseq, max_subseq_steps, max_longseq_steps)
    if return_only_cost is True:
        return cum_sum[..., 0]
    return cum_sum


@njit(cache=True)
def _subsequence_cost_matrix_with_constrains(subseq, longseq, max_subseq_steps, max_longseq_steps):
    # We consider longseq directions as negative and subseq directions as positive values
    max_longseq_steps = -max_longseq_steps
    l1 = subseq.shape[0]
    l2 = longseq.shape[0]
    cum_sum = np.full((l1 + 1, l2 + 1, 2), np.inf)
    cum_sum[0, :] = 0.0
    # All counter values are set to 0
    cum_sum[0:] = 0

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
