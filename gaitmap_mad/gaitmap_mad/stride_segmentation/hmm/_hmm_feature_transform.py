"""Feature transformation class for HMM."""
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn import preprocessing
from tpcp import cf

from gaitmap.data_transform import (
    BaseTransformer,
    ButterworthFilter,
    ChainedTransformer,
    IdentityTransformer,
    ParallelTransformer,
    Resample,
    SlidingWindowGradient,
    SlidingWindowMean,
    SlidingWindowStd,
    SlidingWindowVar,
)
from gaitmap.utils.datatype_helper import SingleSensorData, SingleSensorRegionsOfInterestList

_feature_map = {
    "raw": lambda win_size: IdentityTransformer(),
    "gradient": SlidingWindowGradient,
    "mean": SlidingWindowMean,
    "std": SlidingWindowStd,
    "var": SlidingWindowVar,
}


class FeatureTransformHMM(BaseTransformer):
    """Transform all data and stride labels into the feature space required for training an HMM.

    Default values of all parameters are set based on the work of Nils Roth [1]_

    This applies the following transformations to the data:

    1. Low pass filter followed by a resample to a (lower) sampling rate
    2. On the transformed data the following set of features can be calculated (controlled by `features`):

        - raw (no transformation)
        - gradient (gradient calculated using polyfit on a moving window)
        - mean (sliding window mean)
        - standard deviation (sliding window std)
        - variance (sliding window var)

    The stride labels if provided will simply be transformed to the sampling rate of the feature space.

    This is a very specific high level feature transformer based on the work of Nils Roth [1]_.
    You can create a custom transformer with a similar API interface, if you need more flexibility as exposed by the
    parameters.

    Parameters
    ----------
    sampling_frequency_feature_space_hz
        The sampling rate of the data the model was trained with
    low_pass_cutoff_hz
        Cutoff frequency of low-pass filter for preprocessing
    low_pass_order
        Low-pass filter order
    axis
        List of sensor axis which will be used as model input
    features
        List of features which will be used as model input
    window_size_s
        window size of moving centered window for feature extraction
    standardization
        Flag for feature standardization /  z-score normalization

    Attributes
    ----------
    transformed_data_
        The transformed data.
    transformed_roi_list_
        If a roi_list was provided, this will be the transformed roi list in the new sampling rate

    Other Parameters
    ----------------
    data
        The data passed to the transform method.
    roi_list
        Optional roi list (with values in samples) passed to the transform method
    sampling_rate_hz
        The sampling rate of the input data


    Notes
    -----
    .. [1] Roth, N., Küderle, A., Ullrich, M. et al. Hidden Markov Model based stride segmentation on unsupervised
           free-living gait data in Parkinson’s disease patients. J NeuroEngineering Rehabil 18, 93 (2021).
           https://doi.org/10.1186/s12984-021-00883-7

    """

    # TODO: Find a way to expose the internal objects instead of exposing just parameters.
    sampling_frequency_feature_space_hz: float
    low_pass_cutoff_hz: float
    low_pass_order: int
    axis: List[str]
    features: List[str]
    window_size_s: float
    standardization: bool

    sampling_rate_hz: float

    transformed_roi_list_: SingleSensorRegionsOfInterestList

    def __init__(
        self,
        sampling_frequency_feature_space_hz: float = 51.2,
        low_pass_cutoff_hz: float = 10.0,
        low_pass_order: int = 4,
        axis: List[str] = cf(["gyr_ml"]),
        features: List[str] = cf(["raw", "gradient"]),
        window_size_s: float = 0.2,
        standardization: bool = True,
    ):
        self.sampling_frequency_feature_space_hz = sampling_frequency_feature_space_hz
        self.low_pass_cutoff_hz = low_pass_cutoff_hz
        self.low_pass_order = low_pass_order
        self.axis = axis
        self.features = features
        self.window_size_s = window_size_s
        self.standardization = standardization

    @property
    def n_features(self) -> int:
        return len(self.axis) * len(self.features)

    def transform(
        self,
        data: Optional[SingleSensorData] = None,
        *,
        roi_list: Optional[SingleSensorRegionsOfInterestList] = None,
        sampling_rate_hz: Optional[float] = None,
        **kwargs,
    ):
        """Perform Feature transformation for a single dataset and/or Stride list.

        Parameters
        ----------
        data
            data to be filtered
        roi_list
            Optional roi list/stride list (with values in samples), that will also be resampled to match the data at
            the new sampling rate.
            Note, that only the start and end columns will be modified.
            Other columns remain untouched.
        sampling_rate_hz
            The sampling rate of the data in Hz

        """
        self.sampling_rate_hz = sampling_rate_hz
        if data is not None:
            self.data = data
            if sampling_rate_hz is None:
                raise ValueError(f"{type(self).__name__}.transform requires a `sampling_rate_hz` to be passed.")

            preprocessor = ChainedTransformer(
                [
                    ("filter", ButterworthFilter(self.low_pass_order, self.low_pass_cutoff_hz)),
                    ("resample", Resample(self.sampling_frequency_feature_space_hz)),
                ]
            )
            preprocessor.transform(data, sampling_rate_hz=sampling_rate_hz)

            dataset = preprocessor.transformed_data_

            # All Feature transformers we use work on multiple axis at once.
            # This means, we can just extract the axis we want and throw the result the transformers all at once.
            downsampled_dataset = dataset[self.axis]
            try:
                feature_map = [(k, _feature_map[k](self.window_size_s)) for k in self.features]
            except KeyError as e:
                raise ValueError(
                    "One of the selected features is invalid/not available. "
                    "Available features are:\n"
                    f"{list(_feature_map.keys())}"
                ) from e
            feature_calculator = ParallelTransformer(feature_map)

            # Note we need the sampling rate of the downsampled dataset here
            feature_matrix_df = feature_calculator.transform(
                downsampled_dataset, sampling_rate_hz=self.sampling_frequency_feature_space_hz
            ).transformed_data_
            if self.standardization:
                feature_matrix_df = pd.DataFrame(
                    preprocessing.scale(feature_matrix_df), columns=feature_matrix_df.columns
                )

            self.transformed_data_ = feature_matrix_df
        if roi_list is not None:
            self.transformed_roi_list_ = (
                Resample(self.sampling_frequency_feature_space_hz)
                .transform(roi_list=roi_list, sampling_rate_hz=sampling_rate_hz)
                .transformed_roi_list_
            )

        return self

    def inverse_transform_state_sequence(self, state_sequence: np.ndarray, *, sampling_rate_hz: float) -> np.ndarray:
        """Inverse transform a state sequence to the original sampling rate.

        Parameters
        ----------
        state_sequence
            The state sequence to be transformed back to the original sampling rate.
            This is done by repeating each state for the number of samples it was downsampled to.
        sampling_rate_hz
            The sampling rate of the original data in Hz

        Returns
        -------
        The state sequence in the original sampling rate

        """
        return np.repeat(state_sequence, sampling_rate_hz / self.sampling_frequency_feature_space_hz)

