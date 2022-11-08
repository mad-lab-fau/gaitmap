"""HMM based stride segmentation by Roth et al. 2021."""
import copy
from importlib.resources import open_text
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from tpcp import cf, make_action_safe
from typing_extensions import Self

from gaitmap.base import BaseStrideSegmentation
from gaitmap.stride_segmentation._utils import snap_to_min
from gaitmap.utils._types import _Hashable
from gaitmap.utils.datatype_helper import SensorData, get_multi_sensor_names, is_sensor_data
from gaitmap_mad.stride_segmentation.hmm._segmentation_model import SimpleSegmentationHMM


class PreTrainedRothSegmentationModel(SimpleSegmentationHMM):
    """Load a pre-trained stride segmentation HMM."""

    def __new__(cls):
        # try to load models
        with open_text(
            "gaitmap_mad.stride_segmentation.hmm._pre_trained_models", "fallriskpd_at_lab_model.json"
        ) as test_data:
            with open(test_data.name, encoding="utf8") as f:
                model_json = f.read()
        return SimpleSegmentationHMM.from_json(model_json)


class RothHMM(BaseStrideSegmentation):
    """Segment strides using a pre-trained Hidden Markov Model.

    TBD: short description of HMM

    Parameters
    ----------
    model
        The HMM class need a valid pre-trained model to segment strides
    snap_to_min_win_ms
        The size of the window in seconds used to search local minima during the post processing of the stride borders.
        If this is set to None, this postprocessing step is skipped.
        Refer to the Notes section for more details.
    snap_to_min_axis
        The axis of the data used to search for minima during the processing of the stride borders.
        The axis label must match one of the axis label in the data.
        Refer to the Notes section for more details.

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

    .. [1] ref to JNER HMM paper


    See Also
    --------
    TBD

    """

    snap_to_min_win_ms: float
    snap_to_min_axis: str
    model: Optional[SimpleSegmentationHMM]

    data: Union[np.ndarray, SensorData]
    sampling_rate_hz: float

    matches_start_end_: Union[np.ndarray, Dict[str, np.ndarray]]
    hidden_state_sequence_: Union[np.ndarray, Dict[str, np.ndarray]]
    dataset_feature_space_: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    hidden_state_sequence_feature_space_: Union[np.ndarray, Dict[str, np.ndarray]]

    def __init__(
        self,
        model: SimpleSegmentationHMM = cf(PreTrainedRothSegmentationModel()),
        snap_to_min_win_ms: float = 100,
        snap_to_min_axis: str = "gyr_ml",
    ):
        self.snap_to_min_win_ms = snap_to_min_win_ms
        self.snap_to_min_axis = snap_to_min_axis
        self.model = model

    @property
    def stride_list_(self) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Return start and end of each match as pd.DataFrame."""
        start_ends = self.matches_start_end_
        if isinstance(start_ends, dict):
            return {k: self._format_stride_list(v) for k, v in start_ends.items()}
        return self._format_stride_list(start_ends)

    @property
    def matches_start_end_original_(self) -> Union[np.ndarray, Dict[_Hashable, np.ndarray]]:
        """Return the starts and end directly from the hidden state sequence.

        This will not be effected by potential changes of the postprocessing.
        """
        if isinstance(self.hidden_state_sequence_, dict):
            return {
                s: self._hidden_states_to_matches_start_end(hidden_states)
                for s, hidden_states in self.hidden_state_sequence_.items()
            }
        return self._hidden_states_to_matches_start_end(self.hidden_state_sequence_)

    @staticmethod
    def _format_stride_list(array: np.ndarray) -> pd.DataFrame:
        if len(array) == 0:
            array = None
        as_df = pd.DataFrame(array, columns=["start", "end"])
        # Add the s_id
        as_df.index.name = "s_id"
        return as_df

    @make_action_safe
    def segment(self, data: Union[np.ndarray, SensorData], sampling_rate_hz: float, **_) -> Self:
        """Find matches by predicting a hidden state sequence using a pre-trained Hidden Markov Model.

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

        if dataset_type == "single":
            # Single sensor: easy
            (
                self.matches_start_end_,
                self.hidden_state_sequence_,
                self.dataset_feature_space_,
                self.hidden_state_sequence_feature_space_,
            ) = self._segment_single_dataset(data, sampling_rate_hz)
        else:  # Multisensor
            self.hidden_state_sequence_ = {}
            self.matches_start_end_ = {}
            self.dataset_feature_space_ = {}
            self.hidden_state_sequence_feature_space_ = {}

            for sensor in get_multi_sensor_names(data):
                (
                    matches_start_end,
                    hidden_state_sequence,
                    dataset_feature_space,
                    hidden_state_seq_feature_space,
                ) = self._segment_single_dataset(data[sensor], sampling_rate_hz)
                self.hidden_state_sequence_[sensor] = hidden_state_sequence
                self.matches_start_end_[sensor] = matches_start_end
                self.dataset_feature_space_[sensor] = dataset_feature_space
                self.hidden_state_sequence_feature_space_[sensor] = hidden_state_seq_feature_space

        return self

    def _segment_single_dataset(self, dataset, sampling_rate_hz):
        """Perform Stride Segmentation for a single dataset."""
        # tranform dataset to required feature space as defined by the given model parameters
        model: SimpleSegmentationHMM = self.model.clone()
        feature_data = model.feature_transform.transform(dataset, sampling_rate_hz=sampling_rate_hz).transformed_data_

        hidden_state_sequence = model.predict(feature_data).state_sequence_

        # transform prediction back to original sampling rate!
        downsample_factor = int(
            np.round(sampling_rate_hz / model.feature_transform.sampling_frequency_feature_space_hz)
        )
        hidden_state_sequence_upsampled = np.repeat(hidden_state_sequence, downsample_factor)

        matches_start_end = self._hidden_states_to_matches_start_end(hidden_state_sequence_upsampled)
        return (
            self._postprocess_matches(dataset, matches_start_end),
            hidden_state_sequence_upsampled,
            feature_data,
            hidden_state_sequence,
        )

    def _hidden_states_to_matches_start_end(self, hidden_states_predicted):
        """Convert a hidden state sequence to a list of potential borders."""
        # TODO: Figure out if the strides are inclusive or exclusive the last sample
        stride_start_state = self.model.stride_states_[0]
        stride_end_state = self.model.stride_states_[-1]

        # find rising edge of stride start state sequence
        state_sequence_starts = copy.deepcopy(hidden_states_predicted)
        state_sequence_starts[state_sequence_starts != stride_start_state] = 0
        matches_starts = np.argwhere(np.diff(state_sequence_starts) > 0)

        # find falling edge of stride end state sequence
        state_sequence_ends = copy.deepcopy(hidden_states_predicted)
        state_sequence_ends[state_sequence_ends != stride_end_state] = 0
        matches_ends = np.argwhere(np.diff(state_sequence_ends) < 0)

        # special case where the very last part of the data is just half a stride, so the model finds a begin of a
        # stride but no end! We need to add this end manually
        # TODO: Do we want that? I think we should remove unfinished strides!
        if len(matches_starts) > len(matches_ends):
            matches_ends = np.append(matches_ends, len(hidden_states_predicted))

        return np.column_stack([matches_starts, matches_ends])

    def _postprocess_matches(self, data, matches_start_end) -> np.ndarray:
        """Perform postprocessing step by snapping the stride border candidates to minima within the given data."""
        if self.snap_to_min_win_ms and len(matches_start_end) > 0:
            matches_start_end = snap_to_min(
                data[self.snap_to_min_axis].to_numpy(),
                matches_start_end,
                snap_to_min_win_samples=int(self.snap_to_min_win_ms / 1000 * self.sampling_rate_hz),
            )

        return matches_start_end