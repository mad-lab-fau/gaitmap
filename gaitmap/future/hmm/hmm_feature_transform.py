"""Feature transformation class for HMM."""
import warnings
from typing import Optional, List, Union

import numpy as np
import pandas as pd
from numpy.polynomial import polynomial
from scipy import signal
from sklearn import preprocessing

from gaitmap.base import _BaseSerializable
from gaitmap.utils.array_handling import sliding_window_view
from gaitmap.utils.datatype_helper import get_multi_sensor_names, is_sensor_data


def butterwoth_lowpass(
    data: Union[pd.DataFrame, np.ndarray], fs: float, fc: float, order: int
) -> Union[pd.DataFrame, np.ndarray]:
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


def decimate(data: Union[pd.DataFrame, np.ndarray], downsampling_factor: int) -> Union[pd.DataFrame, np.ndarray]:
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
    - 'polyfit2' : 2nd order polynomial fit for each window
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


class FeatureTransformHMM(_BaseSerializable):
    """Wrap all required information to train a new HMM.

    Parameters
    ----------
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

    sampling_rate_feature_space_hz: Optional[float]
    low_pass_cutoff_hz: Optional[float]
    low_pass_order: Optional[int]
    axis: Optional[List[str]]
    features: Optional[List[str]]
    window_size_s: Optional[int]
    standardization: Optional[bool]

    def __init__(
        self,
        sampling_rate_feature_space_hz: Optional[float] = None,
        low_pass_cutoff_hz: Optional[float] = None,
        low_pass_order: Optional[int] = None,
        axis: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        window_size_s: Optional[int] = None,
        standardization: Optional[bool] = None,
    ):
        self.sampling_rate_feature_space_hz = sampling_rate_feature_space_hz
        self.low_pass_cutoff_hz = low_pass_cutoff_hz
        self.low_pass_order = low_pass_order
        self.axis = axis
        self.features = features
        self.window_size_s = window_size_s
        self.standardization = standardization

    def _transform_single_dataset(self, dataset, sampling_rate_hz):
        """Perform Feature Transformation for a single dataset."""
        is_sensor_data(dataset, check_acc=True, check_gyr=True, frame="body")

        downsample_factor = int(np.round(sampling_rate_hz / self.sampling_rate_feature_space_hz))
        dataset = preprocess(dataset, sampling_rate_hz, self.low_pass_cutoff_hz, self.low_pass_order, downsample_factor)

        window_size_samples = int(np.round(self.window_size_s * self.sampling_rate_feature_space_hz))
        # window size must be odd for centered windows
        if window_size_samples % 2 == 0:
            window_size_samples = window_size_samples + 1

        return calculate_features(dataset, self.axis, self.features, window_size_samples, self.standardization)

    def transform(self, dataset, sampling_rate_hz):
        """Perform a feature transformation according the the given model requirements."""
        dataset_type = is_sensor_data(dataset, check_gyr=False, check_acc=False)

        if dataset_type in ("single", "array"):
            # Single template single sensor: easy
            return self._transform_single_dataset(dataset, sampling_rate_hz)

        # Multisensor
        feature_transformed_dataset = dict()
        for sensor in get_multi_sensor_names(dataset):
            feature_transformed_dataset[sensor] = self._transform_single_dataset(dataset[sensor], sampling_rate_hz)

        return feature_transformed_dataset
