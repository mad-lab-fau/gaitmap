"""A implementation of a sDTW that can be used independent of the context of Stride Segmentation."""
import warnings
from typing import Optional, Sequence, List, Tuple, Union, Dict

import numpy as np
import pandas as pd
from gaitmap.base import BaseType, BaseAlgorithm
from gaitmap.utils.dataset_helper import (
    Dataset,
    is_single_sensor_dataset,
    get_multi_sensor_dataset_names,
    is_dataset,
)


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

    _action_method = "segment"

    template: Optional[DtwTemplate]
    max_cost: Optional[float]
    resample_template: bool
    min_match_length_s: Optional[float]
    max_match_length_s: Optional[float]
    max_template_stretch_ms: Optional[float]
    max_signal_stretch_ms: Optional[float]
    find_matches_method: Literal["min_under_thres", "find_peaks"]

    matches_start_end_: Union[np.ndarray, Dict[str, np.ndarray]]
    acc_cost_mat_: Union[np.ndarray, Dict[str, np.ndarray]]
    paths_: Union[Sequence[Sequence[tuple]], Dict[str, Sequence[Sequence[tuple]]]]
    costs_: Union[Sequence[float], Dict[str, Sequence[float]]]

    data: Union[np.ndarray, Dataset]
    sampling_rate_hz: float

    _allowed_methods_map = {"min_under_thres": find_matches_min_under_threshold, "find_peaks": find_matches_find_peaks}
    _min_sequence_length: Optional[float]
    _max_sequence_length: Optional[float]
    _max_template_stretch: Optional[int]
    _max_signal_stretch: Optional[int]

    def __init__(
        self,
        template: Optional[Union[DtwTemplate, Dict[str, DtwTemplate]]] = None,
        resample_template: bool = True,
        find_matches_method: Literal["min_under_thres", "find_peaks"] = "find_peaks",
        max_cost: Optional[float] = None,
        min_match_length_s: Optional[float] = None,
        max_match_length_s: Optional[float] = None,
        max_template_stretch_ms: Optional[float] = None,
        max_signal_stretch_ms: Optional[float] = None,
    ):
        self.template = template
        self.max_cost = max_cost
        self.min_match_length_s = min_match_length_s
        self.max_match_length_s = max_match_length_s
        self.max_template_stretch_ms = max_template_stretch_ms
        self.max_signal_stretch_ms = max_signal_stretch_ms
        self.resample_template = resample_template
        self.find_matches_method = find_matches_method

    def _validate_basic_inputs(self):
        if self.template is None:
            raise ValueError("A `template` must be specified.")

    def segment(self: BaseType, data: Union[np.ndarray, Dataset], sampling_rate_hz: float, **_) -> BaseType:
        """Find matches by predicting the probability of a hidden state sequence based on a trained HMM.

        Parameters
        ----------
        data : array, single-sensor dataframe, or multi-sensor dataset
            The input data.
            For details on the required datatypes review the class docstring.
        sampling_rate_hz
            The sampling rate of the data signal. This will be used to resample the dataset to the sampling rate of the
            training data for the model.

        Returns
        -------
        self
            The class instance with all result attributes populated

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self._validate_basic_inputs()

        if isinstance(data, np.ndarray):
            dataset_type = "array"
        else:
            dataset_type = is_dataset(data, check_gyr=False, check_acc=False)

        if dataset_type in ("single", "array"):
            # Single template single sensor: easy
            self._segment_single_dataset(data)
        else:  # Multisensor
            for sensor, single_template in template.items():
                print("do multi sensor stuff")
        return self

    def _segment_single_dataset(self, dataset, model):
        # do the magic
        return 0
