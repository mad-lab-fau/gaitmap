"""The msDTW based stride segmentation algorithm by Barth et al 2013."""
from typing import Optional, Union, Dict, List, Tuple, TypeVar
import numpy as np
import pandas as pd
import warnings
from numpy.polynomial import polynomial
from sklearn import preprocessing
from scipy import signal
from gaitmap.base import BaseAlgorithm, BaseType
from gaitmap.stride_segmentation.hmm_models import HiddenMarkovModel
from gaitmap.utils.array_handling import sliding_window_view
from gaitmap.utils.datatype_helper import (
    SensorData,
    get_multi_sensor_names,
    is_sensor_data,
)

array_or_dataframe_type = TypeVar("array_or_dataframe_type", pd.DataFrame, np.ndarray)


def butterwoth_lowpass(data: array_or_dataframe_type, fs: float, fc: float, order: int) -> array_or_dataframe_type:
    """Apply Butterworth Lowpass filter on array or DataFrame."""
    w = fc / (fs / 2)  # Normalize the frequency
    b, a = signal.butter(order, w, "low")

    if fc >= fs:
        warnings.warn("Filter cutoff frequency is >= sampling frequency! Low-pass filter action will be skipped!")
        return data
    if isinstance(data, np.ndarray):
        data_filtered = signal.filtfilt(b, a, data, axis=0)
    if isinstance(data, pd.DataFrame):
        data_filtered = signal.filtfilt(b, a, data.to_numpy(), axis=0)
        data_filtered = pd.DataFrame(data_filtered, columns=data.columns, index=data.index)

    return data_filtered


def decimate(data: array_or_dataframe_type, downsampling_factor: int) -> array_or_dataframe_type:
    """Apply FFT based decimation / downsampling on array or DataFrame."""
    if downsampling_factor <= 1:
        warnings.warn("Downsampling factor <= 1! Downsampling action will be skipped!")
        return data

    if isinstance(data, np.ndarray):
        data_decimated = signal.decimate(data, downsampling_factor, n=None, ftype="iir", axis=0, zero_phase=True)
    if isinstance(data, pd.DataFrame):
        data_decimated = signal.decimate(
            data.to_numpy(),
            downsampling_factor,
            n=None,
            ftype="iir",
            axis=0,
            zero_phase=True,
        )
        data_decimated = pd.DataFrame(data_decimated, columns=data.columns)
    return data_decimated


def preprocess(dataset, fs, fc, filter_order, downsample_factor):
    """Preprocess dataset."""
    dataset_preprocessed = butterwoth_lowpass(dataset, fs, fc, filter_order)
    dataset_preprocessed = decimate(dataset_preprocessed, downsample_factor)
    return dataset_preprocessed


def centered_window_view(arr, window_size_samples, pad_value=0.0):
    """Create a centered window view by zero-padding data before applying sliding window."""
    if not window_size_samples % 2:
        raise ValueError("Window size must be odd for centered windows, but is %d" % window_size_samples)
    pad_length = int(np.floor(window_size_samples / 2))

    if arr.ndim == 1:
        arr = np.pad(arr.astype(float), (pad_length, pad_length), constant_values=pad_value)
    else:
        arr = np.pad(arr.astype(float), [(pad_length, pad_length), (0, 0)], constant_values=pad_value)

    return sliding_window_view(arr, window_size_samples, window_size_samples - 1)


def calculate_features_for_axis(data, window_size_samples, features, standardization=False):
    """Calculate feature matrix on a single sensor axis.

    For feature calculation a centered sliding window with shift of one sample per window is used.

    Available features are:
    - 'raw': raw data itself or "raw data point of window center"
    - 'gradient': first order poly fit = gradient of linear fit within window
    - 'std': standard deviation of each window
    - 'var': variance of each window
    - 'mean': mean of each window
    """
    if window_size_samples > 0:
        window_view = centered_window_view(data, window_size_samples)

    feature_matrix = []
    columns = []

    if not isinstance(features, list):
        features = [features]

    if "raw" in features:
        raw = data.copy()
        feature_matrix.append(raw)
        columns.append("raw")

    if "gradient" in features:
        gradient = np.asarray([polynomial.polyfit(np.arange(window_size_samples), w, 1) for w in window_view])[:, -1]
        feature_matrix.append(gradient)
        columns.append("gradient")

    if "mean" in features:
        mean = window_view.mean(axis=1)
        feature_matrix.append(mean)
        columns.append("mean")

    if "std" in features:
        std = window_view.std(axis=1)
        feature_matrix.append(std)
        columns.append("std")

    if "var" in features:
        var = window_view.var(axis=1)
        feature_matrix.append(var)
        columns.append("var")

    if "polyfit2" in features:
        poly_coefs = np.asarray([polynomial.polyfit(np.arange(window_size_samples), w, 2) for w in window_view])
        feature_matrix.append(poly_coefs)
        columns.append("polyfit2_a")
        columns.append("polyfit2_b")
        columns.append("polyfit2_c")

    feature_matrix = np.column_stack(feature_matrix)

    if standardization:
        feature_matrix = preprocessing.scale(feature_matrix)

    return pd.DataFrame(feature_matrix, columns=columns)


def _calculate_features_single_dataset(dataset, axis, features, window_size_samples, standardization):
    if not isinstance(axis, list):
        axis = [axis]

    axis_feature_dict = {}

    # TODO: Add a check if the requested axis are actually within the given dataset
    for ax in axis:
        # extract features from data
        feature_axis = calculate_features_for_axis(
            dataset[ax].to_numpy(),
            window_size_samples,
            features,
            standardization,
        )
        axis_feature_dict[ax] = feature_axis

    return pd.concat(axis_feature_dict, axis=1)


def calculate_features(dataset, axis, features, window_size_samples, standardization=False):
    """Calculate features on given gaitmap definition dataset.

    data_set: pd.DataFrame
        multisensor dataset in gaitmap bodyframe convention

    Returns
    -------
    feature_dataset: pd.DataFrame
        multicolumn dataset with each column being a calculated feature

    schema of a possible returned pd.DataFrame with 3-level header
        1.-level: sensor
        2.-level: axis
        3.-level: feature

        left_sensor                     right_sensor
        acc_pa          gyr_ml          acc_pa         gyr_ml
        raw     var     raw     var     raw    var     raw      var
        0.1     0.1     0.1     0.1     0.1     0.1     0.1     0.1
        0.1     0.1     0.1     0.1     0.1     0.1     0.1     0.1
        0.1     0.1     0.1     0.1     0.1     0.1     0.1     0.1

    """
    if is_sensor_data(dataset, check_gyr=False, check_acc=False) == "single":
        return _calculate_features_single_dataset(dataset, axis, features, window_size_samples, standardization)

    feature_dict = {}
    for sensor_name in np.unique(dataset.columns.get_level_values(0)):
        feature_dict[sensor_name] = _calculate_features_single_dataset(
            dataset[sensor_name], axis, features, window_size_samples, standardization
        )
    return pd.concat(feature_dict, axis=1)


class RothHMM(BaseAlgorithm):
    """Segment strides using a pre-trained Hidden Markov Model.

    TBD: short description of HMM

    Parameters
    ----------
    model
        The HMM class need a valid pre-trained model to segment strides
    snap_to_min
        Boolean flag to indicate if snap to minimum action shall be performend


    Attributes
    ----------
    stride_list_ : A stride list or dictionary with such values
        The same output as `matches_start_end_`, but as properly formatted pandas DataFrame that can be used as input to
        other algorithms.
        If `snap_to_min` is `True`, the start and end value might not match to the output of `hidden_state_sequence_`.
        Refer to `matches_start_end_original_` for the unmodified start and end values.
    matches_start_end_ : 2D array of shape (n_detected_strides x 2) or dictionary with such values
        The start (column 1) and end (column 2) of each detected stride.
    hidden_state_sequence_ : List of length n_detected_strides or dictionary with such values
        The cost value associated with each stride.
    matches_start_end_original_ : 2D array of shape (n_detected_strides x 2) or dictionary with such values
        Identical to `matches_start_end_` if `snap_to_min` is equal to `False`.
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
        However, this assumes that the start and the end of each match is marked by a clear minimum in one axis of the
        raw data.

    .. [1] ref to JBHI HMM paper


    See Also
    --------
    TBD

    """

    snap_to_min: Optional[bool]
    snap_to_min_axis: Optional[str]
    model: Optional[HiddenMarkovModel]

    data: Union[np.ndarray, SensorData]
    sampling_rate_hz: float

    matches_start_end_: Union[np.ndarray, Dict[str, np.ndarray]]
    hidden_state_sequence_: Union[np.ndarray, Dict[str, np.ndarray]]
    feature_transformed_dataset_: Union[pd.DataFrame, Dict[str, pd.DataFrame]]

    def __init__(
        self,
        model: Optional[HiddenMarkovModel] = None,
        snap_to_min: Optional[bool] = True,
        snap_to_min_axis: Optional[str] = "gyr_ml",
    ):
        self.snap_to_min = snap_to_min
        self.snap_to_min_axis = snap_to_min_axis
        self.model = model

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
        as_df.index.name = "s_id"
        return as_df

    def _postprocess_matches(
        self, data, paths: List, cost: np.ndarray, matches_start_end: np.ndarray, to_keep: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: implement

        # Apply snap to minimum
        if self.snap_to_min:
            return 0

        return 1

    def segment(self: BaseType, data: Union[np.ndarray, SensorData], sampling_rate_hz: float, **_) -> BaseType:
        """Find matches by warping the provided template to the data.

        Parameters
        ----------
        data : array, single-sensor dataframe, or multi-sensor dataset
            The input data.
            For details on the required datatypes review the class docstring.
        sampling_rate_hz
            The sampling rate of the data signal. This will be used to convert all parameters provided in seconds into
            a number of samples and it will be used to perform the required feature transformation`.

        Returns
        -------
        self
            The class instance with all result attributes populated

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        dataset_type = is_sensor_data(data, check_gyr=False, check_acc=False)

        if dataset_type in ("single", "array"):
            # Single sensor: easy
            (
                self.matches_start_end_,
                self.feature_transformed_dataset_,
                self.hidden_state_sequence,
            ) = self._segment_single_dataset(data, sampling_rate_hz)
        else:  # Multisensor
            self.hidden_state_sequence_ = dict()
            self.matches_start_end_ = dict()
            self.feature_transformed_dataset_ = dict()

            for sensor in get_multi_sensor_names(data):
                matches_start_end, feature_transformed_dataset, hidden_state_sequence = self._segment_single_dataset(
                    data[sensor], sampling_rate_hz
                )
                self.hidden_state_sequence_[sensor] = hidden_state_sequence
                self.matches_start_end_[sensor] = matches_start_end
                self.feature_transformed_dataset_[sensor] = feature_transformed_dataset

        return self

    def _segment_single_dataset(self, dataset, sampling_rate_hz):
        """Perform Stride Segmentation for a single dataset"""

        # tranform dataset to required feature space as defined by the given model parameters
        feature_data = self._transform_single_dataset(dataset, sampling_rate_hz)

        hidden_state_sequence = self.model.predict_hidden_states(feature_data, sampling_rate_hz)

        # tranform prediction back to original sampling rate!
        downsample_factor = int(np.round(sampling_rate_hz / self.model.sampling_rate_hz_model))
        hidden_state_sequence = np.repeat(hidden_state_sequence, downsample_factor)

        matches_start_end_ = self._hidden_states_to_stride_borders(
            dataset[self.snap_to_min_axis].to_numpy(), hidden_state_sequence, self.model.stride_states_
        )

        return matches_start_end_, feature_data, hidden_state_sequence

    def _transform_single_dataset(self, dataset, sampling_rate_hz):
        """Perform Feature Transformation for a single dataset"""

        is_sensor_data(dataset, check_acc=True, check_gyr=True, frame="body")

        downsample_factor = int(np.round(sampling_rate_hz / self.model.sampling_rate_hz_model))
        dataset = preprocess(
            dataset, sampling_rate_hz, self.model.low_pass_cutoff_hz, self.model.low_pass_order, downsample_factor
        )

        return calculate_features(
            dataset, self.model.axis, self.model.features, self.model.window_size_samples, self.model.standardization
        )

    def transform(self, dataset, sampling_rate_hz):
        """Perform a feature transformation according the the given model requirements."""
        dataset_type = is_sensor_data(dataset, check_gyr=False, check_acc=False)

        if dataset_type in ("single", "array"):
            # Single template single sensor: easy
            self.feature_transformed_dataset_ = self._transform_single_dataset(dataset, sampling_rate_hz)

        else:  # Multisensor
            self.hidden_state_sequence_, self.matches_start_end_ = dict(), dict()
            for sensor in get_multi_sensor_names(dataset):
                feature_transformed_dataset = self._transform_single_dataset(dataset[sensor], sampling_rate_hz)
                self.feature_transformed_dataset_[sensor] = feature_transformed_dataset

        return self

    def _hidden_states_to_stride_borders(self, data_to_snap_to, hidden_states_predicted, stride_states):
        """This function converts the output of a hmm prediction to meaningful stride borders.

            Therefore, potential stride-borders are derived form the hidden states and the actual border is snapped
            to the data minimum within these potential border windows.
            The potential border windows are derived from the stride-start / end-states plus the adjacent states, which
            might be e.g. transition-stride_start, stride_end-transition, stride_end-stride_start,
            stride_start-stride_end.

        - data_to_snap_to:
            1D array where the "snap to minimum" operation will be performed to find the actual stride border:
            This is usually the "gyr_ml" data!!

        - labels_predicted:
            Predicted hidden-state labels (this should be an array of some discrete values!)

        - stride_states:
            This is the actual list of states we are looking for so e.g. [5,6,7,8,9,10]

        Returns a list of strides with [start,stop] indices

        Example:
        stride_borders = hidden_states_to_stride_borders2(gyr_ml_data, predicted_labels, np.aragne(stride_start_state,
        stride_end_state))
        stride_borders...
        ... [[100,211],
             [211,346],
             [346,478]
             ...
             ]
        """

        if data_to_snap_to.ndim > 1:
            raise ValueError("Snap to minimum only allows 1D arrays as inputs")

        # get all existing label transitions
        transitions, _, _ = self._extract_transitions_starts_stops_from_hidden_state_sequence(hidden_states_predicted)

        if len(transitions) == 0:
            return []

        # START-Window
        adjacent_labels_starts = [a[0] for a in transitions if a[-1] == stride_states[0]]

        potential_start_window = []
        for label in adjacent_labels_starts:
            potential_start_window.extend(
                self._binary_array_to_start_stop_list(
                    np.logical_or(hidden_states_predicted == stride_states[0], hidden_states_predicted == label).astype(
                        int
                    )
                )
            )
        # remove windows where there is actually no stride-start-state label present
        potential_start_window = [
            label
            for label in potential_start_window
            if stride_states[0] in np.unique(hidden_states_predicted[label[0] : label[1] + 1])
        ]

        # melt all windows together
        bin_array_starts = np.zeros(len(hidden_states_predicted))
        for window in potential_start_window:
            bin_array_starts[window[0] : window[1] + 1] = 1

        start_windows = self._binary_array_to_start_stop_list(bin_array_starts)
        start_borders = [np.argmin(data_to_snap_to[window[0] : window[1] + 1]) + window[0] for window in start_windows]

        # END-Window
        adjacent_labels_ends = [
            trans_labels[-1] for trans_labels in transitions if trans_labels[0] == stride_states[-1]
        ]

        potential_end_window = []
        for l in adjacent_labels_ends:
            potential_end_window.extend(
                self._binary_array_to_start_stop_list(
                    np.logical_or(hidden_states_predicted == stride_states[-1], hidden_states_predicted == l).astype(
                        int
                    )
                )
            )

        potential_end_window = [
            a for a in potential_end_window if stride_states[-1] in np.unique(hidden_states_predicted[a[0] : a[1] + 1])
        ]

        # melt all windows together
        bin_array_ends = np.zeros(len(hidden_states_predicted))
        for w in potential_end_window:
            bin_array_ends[w[0] : w[1] + 1] = 1

        end_windows = self._binary_array_to_start_stop_list(bin_array_ends)
        end_borders = [np.argmin(data_to_snap_to[w[0] : w[1] + 1]) + w[0] for w in end_windows]

        return np.array(list(zip(start_borders, end_borders)))

    def _extract_transitions_starts_stops_from_hidden_state_sequence(self, hidden_state_sequence):
        """Return a list of transitions as well as start and stop labels that can be found within the input sequences.

        input = [[1,1,1,1,1,3,3,3,3,2,2,2,2,4,4,4,4,5,5],
                 [0,0,1,1,1,3,3,3,3,2,2,2,6]]
        output_transitions = [[1,3],
                              [3,2],
                              [2,4],
                              [4,5],
                              [0,1],
                              [2,6]]

        output_starts = [1,0]
        output_stops = [5,6]
        """
        if not isinstance(hidden_state_sequence, list):
            hidden_state_sequence = [hidden_state_sequence]

        transitions = []
        starts = []
        ends = []
        for labels in hidden_state_sequence:
            starts.append(labels[0])
            ends.append(labels[-1])
            for idx in np.where(abs(np.diff(labels)) > 0)[0]:
                transitions.append([labels[idx], labels[idx + 1]])

        if len(transitions) > 0:
            transitions = np.unique(transitions, axis=0).astype(int)
        starts = np.unique(starts).astype(int)
        ends = np.unique(ends).astype(int)

        return [transitions, starts, ends]

    def _binary_array_to_start_stop_list(self, bin_array):
        starts = np.where(np.diff(bin_array) > 0)[0] + 1
        stops = np.where(np.diff(bin_array) < 0)[0]
        if bin_array[0] == 1:
            starts = np.insert(starts, 0, 0)
        if bin_array[-1] == 1:
            stops = np.append(stops, len(bin_array) - 1)

        return np.column_stack((starts, stops))
