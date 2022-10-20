"""Feature transformation class for HMM."""
from typing import List, Optional

import pandas as pd
from sklearn import preprocessing

from gaitmap.data_transform import (
    BaseTransformer,
    ButterworthFilter,
    ChainedTransformer,
    Resample,
    IdentityTransformer,
    ParallelTransformer,
    SlidingWindowGradient,
)

_feature_map = {
    "raw": lambda win_size: IdentityTransformer(),
    "gradient": lambda win_size: SlidingWindowGradient(win_size),
}


class FeatureTransformHMM(BaseTransformer):
    """Wrap all required information to train a new HMM.

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

    See Also
    --------
    TBD

    """

    # TODO: Find a way to expose the internal objects instead of exposing just parameters.
    sampling_frequency_feature_space_hz: Optional[float]
    low_pass_cutoff_hz: Optional[float]
    low_pass_order: Optional[int]
    axis: Optional[List[str]]
    features: Optional[List[str]]
    window_size_s: Optional[int]
    standardization: Optional[bool]

    sampling_rate_hz: float

    def __init__(
        self,
        sampling_frequency_feature_space_hz: Optional[float] = None,
        low_pass_cutoff_hz: Optional[float] = None,
        low_pass_order: Optional[int] = None,
        axis: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        window_size_s: Optional[float] = None,
        standardization: Optional[bool] = None,
    ):
        self.sampling_frequency_feature_space_hz = sampling_frequency_feature_space_hz
        self.low_pass_cutoff_hz = low_pass_cutoff_hz
        self.low_pass_order = low_pass_order
        self.axis = axis
        self.features = features
        self.window_size_s = window_size_s
        self.standardization = standardization

    def transform(self, data, *, sampling_rate_hz=None):
        """Perform Feature Transformation for a single dataset."""
        self.sampling_rate_hz = sampling_rate_hz
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
        feature_calculator = ParallelTransformer([(k, _feature_map[k](self.window_size_s)) for k in self.features])
        # Note we need the sampling rate of the downsampled dataset here
        feature_matrix_df = feature_calculator.transform(
            downsampled_dataset, sampling_rate_hz=self.sampling_frequency_feature_space_hz
        ).transformed_data_
        if self.standardization:
            feature_matrix_df = pd.DataFrame(preprocessing.scale(feature_matrix_df), columns=feature_matrix_df.columns)

        self.transformed_data_ = feature_matrix_df

        return self
