"""Dtw template base classes and helper."""
from importlib.resources import open_text
from typing import Optional, List
import json
from pomegranate import HiddenMarkovModel as pgHMM
from gaitmap.base import _BaseSerializable
from typing import TypeVar
import pandas as pd
import numpy as np
from scipy import signal
import warnings
import numpy.polynomial.polynomial as poly
from sklearn import preprocessing

from gaitmap.utils.dataset_helper import is_dataset

from gaitmap.utils.array_handling import sliding_window_view

array_or_dataframe_type = TypeVar("array_or_dataframe_type", pd.DataFrame, np.ndarray)


def butterwoth_lowpass(data: array_or_dataframe_type, fs: float, fc: float, order: int) -> array_or_dataframe_type:
    """Apply Butterworth Lowpass filter on array or DataFrame."""
    w = fc / (fs / 2)  # Normalize the frequency
    b, a = signal.butter(order, w, "low")

    if fc >= fs:
        warnings.warn("Filter cutoff frequency is >= sampling frequency! Low-pass filter action will be skipped!")
        return data
    if isinstance(data, np.ndarray):
        return signal.filtfilt(b, a, data, axis=0)
    if isinstance(data, pd.DataFrame):
        data_filtered = signal.filtfilt(b, a, data.to_numpy(), axis=0)
        return pd.DataFrame(data_filtered, columns=data.columns, index=data.index)


def decimate(data: array_or_dataframe_type, downsampling_factor: int) -> array_or_dataframe_type:
    """Apply FFT based decimation / downsampling on array or DataFrame."""

    if downsampling_factor <= 1:
        warnings.warn("Downsampling factor <= 1! Downsampling action will be skipped!")
        return data

    if isinstance(data, np.ndarray):
        return signal.decimate(data, downsampling_factor, n=None, ftype="iir", axis=0, zero_phase=True)
    if isinstance(data, pd.DataFrame):
        data_decimated = signal.decimate(
            data.to_numpy(), downsampling_factor, n=None, ftype="iir", axis=0, zero_phase=True,
        )
        return pd.DataFrame(data_decimated, columns=data.columns)


def preprocess(dataset, fs, fc, filter_order, downsample_factor):
    dataset_preprocessed = butterwoth_lowpass(dataset, fs, fc, filter_order)
    dataset_preprocessed = decimate(dataset_preprocessed, downsample_factor)
    return dataset_preprocessed


def centered_window_view(arr, window_size_samples, pad_value=0.0):
    """Create a centered window view by zero-padding data before applying sliding window"""

    if not (window_size_samples % 2):
        raise ValueError("Window size must be odd for centered windows, but is %d" % window_size_samples)
    pad_length = int(np.floor(window_size_samples / 2))

    if arr.ndim == 1:
        arr = np.pad(arr.astype(float), (pad_length, pad_length), constant_values=pad_value)
    else:
        arr = np.pad(arr.astype(float), [(pad_length, pad_length), (0, 0)], constant_values=pad_value)

    return sliding_window_view(arr, window_size_samples, window_size_samples - 1)


def calculate_features_for_axis(data, window_size_samples, features, standardization=False):
    """Calculate feature matrix on a single sensor axis. For feature calculation a centered sliding window with shift
    of one sample per window is used.

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
        gradient = np.asarray([poly.polyfit(np.arange(window_size_samples), w, 1) for w in window_view])[:, -1]
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
        poly_coefs = np.asarray([poly.polyfit(np.arange(window_size_samples), w, 2) for w in window_view])
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
    for axis in axis:
        # extract features from data
        feature_axis = calculate_features_for_axis(
            dataset[axis].to_numpy(), window_size_samples, features, standardization,
        )
        axis_feature_dict[axis] = feature_axis

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

    left_sensor 				 	right_sensor
    acc_pa	 		gyr_ml 		 	acc_pa	 		gyr_ml
    raw		var 	raw		var	 	raw		var 	raw		var
    0.1	    0.1     0.1     0.1     0.1     0.1     0.1     0.1
    0.1	    0.1     0.1     0.1     0.1     0.1     0.1     0.1
    0.1	    0.1     0.1     0.1     0.1     0.1     0.1     0.1
     .       .       .       .       .       .       .       .
     .       .       .       .       .       .       .       .
     .       .       .       .       .       .       .       .
    """
    if is_dataset(dataset, check_gyr=False, check_acc=False) is "single":
        return _calculate_features_single_dataset(dataset, axis, features, window_size_samples, standardization)
    else:
        feature_dict = {}
        for sensor_name in np.unique(dataset.columns.get_level_values(0)):
            feature_dict[sensor_name] = _calculate_features_single_dataset(
                dataset[sensor_name], axis, features, window_size_samples, standardization
            )
        return pd.concat(feature_dict, axis=1)


class HiddenMarkovModel(_BaseSerializable):
    """Wrap all required information about a pre-trained HMM.

    Parameters
    ----------
    model_file_name
        Path to a valid pre-trained and serialized model-json
    sampling_rate_hz_model
        The sampling rate of the data the model was trained with
    low_pass_cutoff_hz
        Cutoff frequency of low-pass filter for preprocessing
    low_pass_order
        Low-pass filter order
    axis
        List of sensor axis which will be used as model input
    features
        List of features which will be used as model input
    window_size_samples
        window size of moving centered window for feature extraction
    standardization
        Flag for feature standardization /  z-score normalization

    See Also
    --------
    TBD

    """

    model_file_name: Optional[str]
    sampling_rate_hz_model: Optional[float]
    low_pass_cutoff_hz: Optional[float]
    low_pass_order: Optional[int]
    axis: Optional[List[str]]
    features: Optional[List[str]]
    window_size_samples: Optional[int]
    standardization: Optional[bool]

    _model_combined: Optional[pgHMM]
    _model_stride: Optional[pgHMM]
    _model_transition: Optional[pgHMM]
    _n_states_stride: Optional[int]
    _n_states_transition: Optional[int]

    def __init__(
        self,
        model_file_name: Optional[str] = None,
        sampling_rate_hz_model: Optional[float] = None,
        low_pass_cutoff_hz: Optional[float] = None,
        low_pass_order: Optional[int] = None,
        axis: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        window_size_samples: Optional[int] = None,
        standardization: Optional[bool] = None,
    ):
        self.model_file_name = model_file_name
        self.sampling_rate_hz_model = sampling_rate_hz_model
        self.low_pass_cutoff_hz = low_pass_cutoff_hz
        self.low_pass_order = low_pass_order
        self.axis = axis
        self.features = features
        self.window_size_samples = window_size_samples
        self.standardization = standardization

        # try to load models
        with open_text("gaitmap.stride_segmentation.hmm_models", self.model_file_name) as test_data:
            with open(test_data.name) as f:
                models_dict = json.load(f)

        if (
            "combined_model" not in models_dict.keys()
            or "stride_model" not in models_dict.keys()
            or "transition_model" not in models_dict.keys()
        ):
            raise ValueError(
                "Invalid model-json! Keys within the model-json required are: 'combined_model', 'stride_model' and 'transition_model'"
            )

        self._model_stride = pgHMM.from_json(models_dict["stride_model"])
        self._model_transition = pgHMM.from_json(models_dict["transition_model"])
        self._model_combined = pgHMM.from_json(models_dict["combined_model"])

        # we need to subtract the silent start and end state form the state count!
        self._n_states_stride = self._model_stride.state_count() - 2
        self._n_states_transition = self._model_transition.state_count() - 2

    def transform(self, dataset, sampling_rate_hz):
        """Perform a feature transformation according the the given model requirements."""

        # check if dataset is in body frame notation
        is_dataset(dataset, check_acc=True, check_gyr=True, frame="body")

        downsample_factor = int(np.round(sampling_rate_hz / self.sampling_rate_hz_model))
        dataset = preprocess(dataset, sampling_rate_hz, self.low_pass_cutoff_hz, self.low_pass_order, downsample_factor)

        return calculate_features(dataset, self.axis, self.features, self.window_size_samples, self.standardization)

    def predict(self, dataset, sampling_rate_hz, algorithm="viterbi"):

        feature_data = self.transform(dataset, sampling_rate_hz)

        hidden_state_sequence = predict_hidden_states(
            self._model_combined, np.ascontiguousarray(feature_data.to_numpy()), algorithm
        )

        # tranform prediction back to original sampling rate!
        downsample_factor = int(np.round(sampling_rate_hz / self.sampling_rate_hz_model))
        return np.repeat(hidden_state_sequence, downsample_factor)

    @property
    def stride_states_(self):
        return np.arange(self._n_states_transition, self._n_states_stride + self._n_states_transition)

    @property
    def transition_states_(self):
        return np.arange(self._n_states_transition)

    def _predict_hidden_states(self, model, feature_data, algorithm):
        """Perform prediction based on given data and given model."""
        # need to check if memory layout of given data is
        # see related pomegranate issue: https://github.com/jmschrei/pomegranate/issues/717
        if not np.array(feature_data).flags["C_CONTIGUOUS"]:
            raise ValueError("Memory Layout of given input data is not contiguois! Consider using ")

        labels_predicted = np.asarray(model.predict(feature_data, algorithm=algorithm))
        # pomegranate always adds an additional label for the start- and end-state, which can be ignored here!
        return np.asarray(labels_predicted[1:-1])


class HiddenMarkovModelStairs(HiddenMarkovModel):
    """Hidden Markov Model trained for stride segmentation including stair strides.

    Notes
    -----
    This is a pre-trained model aiming to segment also "stair strides" next to "normal strides"

    See Also
    --------
    TBD

    """

    model_file_name = "hmm_stairs.json"
    # preprocessing settings
    sampling_rate_hz_model = 51.2
    low_pass_cutoff_hz = 10.0
    low_pass_order = 4
    # feature settings
    axis = ["gyr_ml"]
    features = ["raw", "gradient"]
    window_size_samples = 11
    standardization = True

    def __init__(self):
        super().__init__(
            model_file_name=self.model_file_name,
            sampling_rate_hz_model=self.sampling_rate_hz_model,
            low_pass_cutoff_hz=self.low_pass_cutoff_hz,
            low_pass_order=self.low_pass_order,
            axis=self.axis,
            features=self.features,
            window_size_samples=self.window_size_samples,
            standardization=self.standardization,
        )
