"""A implementation of a sDTW that can be used independent of the context of Stride Segmentation."""

from typing import Optional, Sequence, List, Tuple, Union, Dict

import numpy as np
import pandas as pd
from scipy.signal import resample
from tslearn.metrics import subsequence_cost_matrix, subsequence_path
from tslearn.utils import to_time_series
from typing_extensions import Literal

from gaitmap.base import BaseStrideSegmentation, BaseType
from gaitmap.stride_segmentation.dtw_templates import DtwTemplate
from gaitmap.stride_segmentation.utils import find_local_minima_with_distance, find_local_minima_below_threshold
from gaitmap.utils.dataset_helper import (
    Dataset,
    is_single_sensor_dataset,
    is_multi_sensor_dataset,
    get_multi_sensor_dataset_names,
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

    TODO: Add link to dataset doumentation in the future


    Parameters
    ----------
    template
        The template used for matching.
        The required data type and shape depends on the use case.
        For more details view the class docstring.
    resample_template
        If `True` the template will be resampled to match the sampling rate of the data.
        This requires a valid value for `template.sampling_rate_hz` value.
    max_cost
        The maximal allowed cost to find potential match in the cost function.
        Its usage depends on the exact `find_matches_method` used.
        Refer to the specific funtion to learn more about this.
    min_match_length
        The minimal length of a sequence in samples to be considered a match.
        Matches that result in shorter sequences, will be ignored.
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

    Attributes
    ----------
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
    min_match_length: Optional[int]
    find_matches_method: Literal["min_under_thres", "find_peaks"]

    matches_start_end_: Union[np.ndarray, Dict[str, np.ndarray]]
    acc_cost_mat_: Union[np.ndarray, Dict[str, np.ndarray]]
    paths_: Union[Sequence[Sequence[tuple]], Dict[str, Sequence[Sequence[tuple]]]]
    costs_: Union[Sequence[float], Dict[str, Sequence[float]]]

    data: Union[np.ndarray, Dataset]
    sampling_rate_hz: float

    _allowed_methods_map = {"min_under_thres": find_matches_min_under_threshold, "find_peaks": find_matches_find_peaks}

    @property
    def cost_function_(self):
        """Cost function extracted from the accumulated cost matrix."""
        if isinstance(self.acc_cost_mat_, dict):
            return {s: np.sqrt(cost_mat[-1, :]) for s, cost_mat in self.acc_cost_mat_.items()}
        return np.sqrt(self.acc_cost_mat_[-1, :])

    def __init__(
        self,
        template: Optional[Union[DtwTemplate, Dict[str, DtwTemplate]]] = None,
        resample_template: bool = True,
        find_matches_method: Literal["min_under_thres", "find_peaks"] = "find_peaks",
        max_cost: Optional[int] = None,
        min_match_length: Optional[int] = None,
    ):
        self.template = template
        self.max_cost = max_cost
        self.min_match_length = min_match_length
        self.resample_template = resample_template
        self.find_matches_method = find_matches_method

    def segment(self: BaseType, data: Union[np.ndarray, Dataset], sampling_rate_hz: float, **_) -> BaseType:
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

        # Validate and transform inputs
        if self.template is None:
            raise ValueError("A `template` must be specified.")

        template = self.template
        if isinstance(data, np.ndarray) or is_single_sensor_dataset(data, check_gyr=False, check_acc=False):
            # Single template single sensor: easy
            self.acc_cost_mat_, self.paths_, self.costs_, self.matches_start_end_ = self._segment_single_dataset(
                data, template
            )
        elif is_multi_sensor_dataset(data, check_gyr=False, check_acc=False):
            if isinstance(template, dict):
                # multiple templates, multiple sensors: Apply the correct template to the correct sensor.
                # Ignore the rest
                results = dict()
                for sensor, single_template in template.items():
                    results[sensor] = self._segment_single_dataset(data[sensor], single_template)
            elif is_single_sensor_dataset(template.template, check_gyr=False, check_acc=False):
                # single template, multiple sensors: Apply template to all sensors
                results = dict()
                for sensor in get_multi_sensor_dataset_names(data):
                    results[sensor] = self._segment_single_dataset(data[sensor], template)
            else:
                # TODO: Test
                raise ValueError(
                    "In case of a multi-sensor dataset input, the used template must either be of type "
                    "`Dict[str, DtwTemplate]` or the template array must have the shape of a single-sensor dataframe."
                )
            self.acc_cost_mat_, self.paths_, self.costs_, self.matches_start_end_ = dict(), dict(), dict(), dict()
            for sensor, r in results.items():
                self.acc_cost_mat_[sensor] = r[0]
                self.paths_[sensor] = r[1]
                self.costs_[sensor] = r[2]
                self.matches_start_end_[sensor] = r[3]
        else:
            # TODO: Better error message
            # TODO: Test
            raise ValueError("The type or shape of the provided dataset is not supported.")
        return self

    def _segment_single_dataset(self, dataset, template):
        if self.resample_template and not template.sampling_rate_hz:
            raise ValueError(
                "To resample the template (`resample_template=True`), a `sampling_rate_hz` must be specified for the "
                "template."
            )

        # Extract the parts of the data that is relevant for matching.
        template_array, matching_data = self._extract_relevant_data_and_template(template.template, dataset)

        if self.resample_template is True and self.sampling_rate_hz != template.sampling_rate_hz:
            template = self._resample_template(template_array, template.sampling_rate_hz, self.sampling_rate_hz)
        else:
            template = template_array

        min_distance = self.min_match_length

        find_matches_method = self._allowed_methods_map.get(self.find_matches_method, None)
        if not find_matches_method:
            raise ValueError(
                'Invalid value for "find_matches_method". Must be one of {}'.format(
                    list(self._allowed_methods_map.keys())
                )
            )

        # Calculate cost matrix
        acc_cost_mat_ = subsequence_cost_matrix(to_time_series(template), to_time_series(matching_data))

        matches = find_matches_method(acc_cost_mat=acc_cost_mat_, max_cost=self.max_cost, min_distance=min_distance)
        if len(matches) == 0:
            paths_ = []
            costs_ = []
            matches_start_end_ = []
        else:
            paths_ = self._find_multiple_paths(acc_cost_mat_, matches)
            matches_start_end_ = np.array([[p[0][-1], p[-1][-1]] for p in paths_])

            # Remove matches that are shorter that min_match_length
            if min_distance is None:
                min_distance = -np.inf
            valid_strides = np.squeeze(np.abs(np.diff(matches_start_end_, axis=-1)) > min_distance)
            valid_strides_idx = np.where(valid_strides)[0]
            matches_start_end_ = matches_start_end_[valid_strides_idx]
            paths_ = [paths_[i] for i in valid_strides_idx]
            costs_ = np.sqrt(acc_cost_mat_[-1, :][matches[valid_strides]])

        return acc_cost_mat_, paths_, costs_, matches_start_end_

    @staticmethod
    def _resample_template(
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
        # TODO: Test
        raise ValueError("Invalid combination of data and template")

    @staticmethod
    def _find_multiple_paths(acc_cost_mat: np.ndarray, start_points: np.ndarray) -> List[np.ndarray]:
        paths = []
        for start in start_points:
            path = subsequence_path(acc_cost_mat, start)
            path_array = np.array(path)
            paths.append(path_array)
        return paths
