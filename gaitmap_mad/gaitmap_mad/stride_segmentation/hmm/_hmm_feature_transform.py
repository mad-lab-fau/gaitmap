"""Feature transformation class for HMM."""

from typing import NoReturn, Optional

import numpy as np
import pandas as pd
from sklearn import preprocessing
from tpcp import cf

from gaitmap.data_transform import (
    BaseFilter,
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
    "raw": lambda _: IdentityTransformer(),
    "gradient": SlidingWindowGradient,
    "mean": SlidingWindowMean,
    "std": SlidingWindowStd,
    "var": SlidingWindowVar,
}


class BaseHmmFeatureTransformer(BaseTransformer):
    """Baseclass for HMM feature transformers used in combination with `SimpleSegmentationModel`.

    This is only required if :class:`gaitmap.stride_segmentation.hmm.RothHMMFeatureTransformer`
    is not sufficient for your use case, when using :class:`~gaitmap.stride_segmentation.hmm.RothSegmentationHmm`.

    In this case implement a custom subclass and pass it to the `feature_transform` parameter of
    `RothSegmentationHmm`.
    Note, that you need to implement the `transform` and `inverse_transform_state_sequence` methods.
    """

    sampling_rate_hz: float
    roi_list: SingleSensorRegionsOfInterestList

    transformed_roi_list_: SingleSensorRegionsOfInterestList

    def transform(
        self,
        data: Optional[SingleSensorData] = None,
        *,
        roi_list: Optional[SingleSensorRegionsOfInterestList] = None,
        sampling_rate_hz: Optional[float] = None,
        **kwargs,
    ) -> NoReturn:
        """Transform the data and the roi/stride list into to the feature space.

        Transforming the roi/stride list is only required, if the sampling rate of the features space is different from
        the data space.
        If now down-sampling is required, set `self.transformed_roi_list_` to `roi_list`.

        """
        raise NotImplementedError()

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
        raise NotImplementedError()


class RothHmmFeatureTransformer(BaseHmmFeatureTransformer):
    """Transform all data and stride labels into the feature space required for training an HMM.

    This method is expected to be used in combination with the
    :class:`~gaitmap.stride_segmentation.hmm.RothSegmentationHmm` class.
    Default values of all parameters are set based on the work of Nils Roth [1]_.

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
    parameters (see :class:`~gait.gaitmap.stride_segmentation.hmm.BaseHmmFeatureTransformer`).

    Parameters
    ----------
    sampling_rate_feature_space_hz
        The sampling rate of the data the model was trained with
    low_pass_filter
        Instance of a low pass filter to be applied to the data before resampling.
        Note, that this filter is not strictly required, as the downsampling will apply a second filter to ensure
        that the Nyquist frequency is not exceeded.
        However, you might want to use this filter to smooth the signal beyond what is required for correct
        downsampling.
        Can be disabled by setting to `None`.
    axes
        List of sensor axes which will be used as model input
    features
        List of features which will be used as model input
    window_size_s
        window size of moving centered window for feature extraction
    standardization
        Flag for feature standardization /  z-score normalization.
        Note, that this is done individually for "train" and "test" data.
        I.e., we don't store the standardization parameters from the "train" data.
        In the particular case of the Roth HMM this worked well, as sufficient data was available for testing.

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
           free-living gait data in Parkinson`s disease patients. J NeuroEngineering Rehabil 18, 93 (2021).
           https://doi.org/10.1186/s12984-021-00883-7

    """

    # TODO: Find a way to expose the internal objects instead of exposing just parameters.
    sampling_rate_feature_space_hz: float
    low_pass_filter: Optional[BaseFilter]
    axes: list[str]
    features: list[str]
    window_size_s: float
    standardization: bool

    def __init__(
        self,
        sampling_rate_feature_space_hz: float = 51.2,
        low_pass_filter: Optional[BaseFilter] = cf(ButterworthFilter(cutoff_freq_hz=10.0, order=4)),
        axes: list[str] = cf(["gyr_ml"]),
        features: list[str] = cf(["raw", "gradient"]),
        window_size_s: float = 0.2,
        standardization: bool = True,
    ) -> None:
        self.sampling_rate_feature_space_hz = sampling_rate_feature_space_hz
        self.low_pass_filter = low_pass_filter
        self.axes = axes
        self.features = features
        self.window_size_s = window_size_s
        self.standardization = standardization

    @property
    def n_features(self) -> int:
        """Get the number of features in the transformed data."""
        return len(self.axes) * len(self.features)

    def transform(
        self,
        data: Optional[SingleSensorData] = None,
        *,
        roi_list: Optional[SingleSensorRegionsOfInterestList] = None,
        sampling_rate_hz: Optional[float] = None,
        **_,
    ):
        """Perform feature transformation for a single dataset and/or stride list.

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
        if sampling_rate_hz is None:
            raise ValueError(f"{type(self).__name__}.transform requires a `sampling_rate_hz` to be passed.")
        if data is not None:
            self.data = data

            if self.low_pass_filter is not None and not isinstance(self.low_pass_filter, BaseFilter):
                raise TypeError(f"{type(self).__name__}.low_pass_filter must be a subclass of BaseFilter.")

            preprocessor = ChainedTransformer(
                [
                    ("filter", (self.low_pass_filter or IdentityTransformer()).clone()),
                    ("resample", Resample(self.sampling_rate_feature_space_hz)),
                ]
            )
            preprocessor.transform(data, sampling_rate_hz=sampling_rate_hz)

            dataset = preprocessor.transformed_data_

            # All Feature transformers we use work on multiple axis at once.
            # This means, we can just extract the axis we want and throw the result the transformers all at once.
            downsampled_dataset = dataset[self.axes]
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
                downsampled_dataset, sampling_rate_hz=self.sampling_rate_feature_space_hz
            ).transformed_data_
            if self.standardization:
                feature_matrix_df = pd.DataFrame(
                    preprocessing.scale(feature_matrix_df), columns=feature_matrix_df.columns
                )

            self.transformed_data_ = feature_matrix_df
        if roi_list is not None:
            self.roi_list = roi_list
            self.transformed_roi_list_ = (
                Resample(self.sampling_rate_feature_space_hz)
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
        return np.repeat(state_sequence, sampling_rate_hz / self.sampling_rate_feature_space_hz)
