"""HMM based stride segmentation by Roth et al. 2021."""
from importlib.resources import open_text
from typing import Dict, Generic, TypeVar, Union

import numpy as np
import pandas as pd
from tpcp import cf, make_action_safe
from typing_extensions import Self

from gaitmap.base import BaseStrideSegmentation
from gaitmap.stride_segmentation._utils import snap_to_min
from gaitmap.utils._types import _Hashable
from gaitmap.utils.datatype_helper import SensorData, get_multi_sensor_names, is_sensor_data
from gaitmap_mad.stride_segmentation.hmm._segmentation_model import BaseSegmentationHmm, RothSegmentationHmm


class PreTrainedRothSegmentationModel(RothSegmentationHmm):
    """Load a pre-trained stride segmentation HMM."""

    def __new__(cls):
        # try to load models
        with open_text(
            "gaitmap_mad.stride_segmentation.hmm._pre_trained_models", "fallriskpd_at_lab_model.json"
        ) as test_data:
            with open(test_data.name, encoding="utf8") as f:
                model_json = f.read()
        return RothSegmentationHmm.from_json(model_json)


BaseSegmentationHmmT = TypeVar("BaseSegmentationHmmT", bound=BaseSegmentationHmm)


class HmmStrideSegmentation(BaseStrideSegmentation, Generic[BaseSegmentationHmmT]):
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
    matches_start_end_ : 2D array of shape (n_detected_strides x 2) or dictionary with such values
        The start (column 1) and end (column 2) of each detected stride.
    stride_list_ : A stride list or dictionary with such values
        The same output as `matches_start_end_`, but as properly formatted pandas DataFrame that can be used as input to
        other algorithms.
        If `snap_to_min` is `True`, the start and end value might not match to the output of `hidden_state_sequence_`.
        Refer to `matches_start_end_original_` for the unmodified start and end values.
    matches_start_end_original_ : 2D array of shape (n_detected_strides x 2) or dictionary with such values
        Identical to `matches_start_end_` if `snap_to_min` is equal to `False`.
        Otherwise, it returns the start and end values before the snapping is applied.
    hidden_state_sequence_ : List of length n_detected_strides or dictionary with such values
        The cost value associated with each stride.
    result_models_
        The copy of the model used for the segmentation with all the result parameters attached.


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
    model: BaseSegmentationHmmT

    data: Union[np.ndarray, SensorData]
    sampling_rate_hz: float

    matches_start_end_: Union[np.ndarray, Dict[str, np.ndarray]]
    hidden_state_sequence_: Union[np.ndarray, Dict[str, np.ndarray]]
    result_models_: Union[BaseSegmentationHmmT, Dict[str, BaseSegmentationHmmT]]

    def __init__(
        self,
        model: BaseSegmentationHmmT = cf(PreTrainedRothSegmentationModel()),
        *,
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
    def segment(self, data: SensorData, sampling_rate_hz: float, **_) -> Self:
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
                self.result_models_,
            ) = self._segment_single_dataset(data, sampling_rate_hz=sampling_rate_hz)
        else:  # Multisensor
            self.hidden_state_sequence_ = {}
            self.matches_start_end_ = {}
            self.result_models_ = {}

            for sensor in get_multi_sensor_names(data):
                (
                    matches_start_end,
                    hidden_state_sequence,
                    result_model,
                ) = self._segment_single_dataset(data[sensor], sampling_rate_hz=sampling_rate_hz)
                self.hidden_state_sequence_[sensor] = hidden_state_sequence
                self.matches_start_end_[sensor] = matches_start_end
                self.result_models_[sensor] = result_model

        return self

    def _segment_single_dataset(self, dataset, *, sampling_rate_hz: float):
        """Perform Stride Segmentation for a single dataset."""
        model: BaseSegmentationHmm = self.model.clone()
        model = model.predict(dataset, sampling_rate_hz=sampling_rate_hz)
        state_sequence = model.hidden_state_sequence_

        matches_start_end = self._hidden_states_to_matches_start_end(state_sequence)
        return (
            self._postprocess_matches(dataset, matches_start_end),
            state_sequence,
            model,
        )

    def _hidden_states_to_matches_start_end(self, hidden_states_predicted: np.ndarray):
        """Convert a hidden state sequence to a list of potential borders."""
        # TODO: Figure out if the strides are inclusive or exclusive the last sample
        stride_start_state = self.model.stride_states[0]
        stride_end_state = self.model.stride_states[-1]

        # find rising edge of stride start state sequence
        matches_starts = np.argwhere(np.diff((hidden_states_predicted == stride_start_state).astype(int)) > 0)

        # find falling edge of stride end state sequence
        matches_ends = np.argwhere(np.diff((hidden_states_predicted == stride_end_state).astype(int)) < 0)

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
