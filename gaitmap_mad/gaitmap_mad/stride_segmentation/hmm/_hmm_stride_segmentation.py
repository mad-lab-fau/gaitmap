"""HMM based stride segmentation by Roth et al. 2021."""
from contextlib import suppress
from typing import Dict, Generic, Optional, TypeVar, Union

import numpy as np
import pandas as pd
from tpcp import cf, make_action_safe
from typing_extensions import Self

from gaitmap.base import BaseStrideSegmentation
from gaitmap.stride_segmentation._utils import snap_to_min
from gaitmap.utils._algo_helper import invert_result_dictionary, set_params_from_dict
from gaitmap.utils._types import _Hashable
from gaitmap.utils.datatype_helper import SensorData, get_multi_sensor_names, is_sensor_data
from gaitmap_mad.stride_segmentation.hmm._segmentation_model import BaseSegmentationHmm, RothSegmentationHmm


class PreTrainedRothSegmentationModel(RothSegmentationHmm):
    """Load a pre-trained stride segmentation HMM.

    Notes
    -----
    This model was trained on the pre-visit @lab recordings of the first 28 participants of the fallrisk-pd study.
    According to [1]_ the expected performance on unseen data under lab conditions is around 96% F1 score and under
    real-world conditions ca. 92% F1 score.

    The model is only for level walking and was only trained on PD data (so it might not generalize well to other
    conditions).

    Recommended use for general segmentation of straight strides.
    But, the model will probably also segment turning strides as it only considers the `gyr_ml` data.
    If only straight strides are desired, strides should be filtered based on turning angle after parameter estimation.

    .. [1] Roth, N., Küderle, A., Ullrich, M. et al. Hidden Markov Model based stride segmentation on unsupervised
           free-living gait data in Parkinson`s disease patients. J NeuroEngineering Rehabil 18, 93 (2021).
           https://doi.org/10.1186/s12984-021-00883-7

    """

    # def __new__(cls):
    #     # try to load models
    #     with open_text(
    #         "gaitmap_mad.stride_segmentation.hmm._pre_trained_models", "fallriskpd_at_lab_model.json"
    #     ) as test_data, Path(test_data.name).open(encoding="utf8") as f:
    #         model_json = f.read()
    #     return RothSegmentationHmm.from_json(model_json)


BaseSegmentationHmmT = TypeVar("BaseSegmentationHmmT", bound=BaseSegmentationHmm)


class HmmStrideSegmentation(BaseStrideSegmentation, Generic[BaseSegmentationHmmT]):
    """Segment strides using a pre-trained Hidden Markov Model.

    This method does not care about the implementation details of the HMM used.
    As long as it is a valid subclass of BaseSegmentationHmm, it can be used here.
    On top of the segmentation, this class implements a postprocessing step that snaps the minima in the raw signals
    and contains convenience methods that ensure that outputs conform to the expected gaitmap formats.

    Note, that this class only supports prediction.
    To train your own HMM, use the `self_optimize` method on the model that you were planning to use here.

    This is based on the work of Roth et al. 2021 [1]_ and the implementation is using done with `pomegranate` [2]_.

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
    result_model_
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

    .. [1] Roth, N., Küderle, A., Ullrich, M. et al. Hidden Markov Model based stride segmentation on unsupervised
       free-living gait data in Parkinson`s disease patients. J NeuroEngineering Rehabil 18, 93 (2021).
       https://doi.org/10.1186/s12984-021-00883-7
    .. [2] J. Schreiber, “Pomegranate: Fast and flexible probabilistic modeling in python,” Journal of Machine Learning
       Research, vol. 18, no. 164, pp. 1-6, 2018.

    """

    snap_to_min_win_ms: Optional[float]
    snap_to_min_axis: str
    model: BaseSegmentationHmmT

    data: Union[np.ndarray, SensorData]
    sampling_rate_hz: float

    matches_start_end_: Union[np.ndarray, Dict[str, np.ndarray]]
    hidden_state_sequence_: Union[np.ndarray, Dict[str, np.ndarray]]
    result_model_: Union[BaseSegmentationHmmT, Dict[str, BaseSegmentationHmmT]]

    def __init__(
        self,
        model: BaseSegmentationHmmT = cf(PreTrainedRothSegmentationModel()),
        *,
        snap_to_min_win_ms: Optional[float] = 100,
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
        stride_start_state = self.model.stride_states[0]
        stride_end_state = self.model.stride_states[-1]
        if isinstance(self.hidden_state_sequence_, dict):
            return {
                s: self._hidden_states_to_matches_start_end(hidden_states, stride_start_state, stride_end_state)
                for s, hidden_states in self.hidden_state_sequence_.items()
            }

        return self._hidden_states_to_matches_start_end(
            self.hidden_state_sequence_, stride_start_state, stride_end_state
        )

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
            results = self._segment_single_dataset(data, sampling_rate_hz=sampling_rate_hz)
        else:  # Multisensor
            result_dict = {
                sensor: self._segment_single_dataset(data[sensor], sampling_rate_hz=sampling_rate_hz)
                for sensor in get_multi_sensor_names(data)
            }
            results = invert_result_dictionary(result_dict)
        set_params_from_dict(self, results, result_formatting=True)
        return self

    def _segment_single_dataset(self, dataset, *, sampling_rate_hz: float):
        """Perform Stride Segmentation for a single dataset."""
        model: BaseSegmentationHmm = self.model.clone()
        model = model.predict(dataset, sampling_rate_hz=sampling_rate_hz)
        state_sequence = model.hidden_state_sequence_
        stride_start_state = self.model.stride_states[0]
        stride_end_state = self.model.stride_states[-1]
        matches_start_end = self._hidden_states_to_matches_start_end(
            state_sequence, stride_start_state, stride_end_state
        )
        return {
            "matches_start_end": self._postprocess_matches(dataset, matches_start_end),
            "hidden_state_sequence": state_sequence,
            "result_model_": model,
        }

    def _hidden_states_to_matches_start_end(
        self, hidden_states_predicted: np.ndarray, stride_start_state, stride_end_state
    ):
        """Convert a hidden state sequence to a list of potential borders."""
        # find rising edge of stride start state sequence
        # +1 required as diff returns one element less than the input
        matches_starts = (
            np.argwhere(np.diff((hidden_states_predicted == stride_start_state).astype("int64")) > 0).flatten() + 1
        )

        # find falling edge of stride end state sequence
        matches_ends = (
            np.argwhere(np.diff((hidden_states_predicted == stride_end_state).astype("int64")) < 0).flatten() + 1
        )

        # Special case, when the last state is a stride end state
        if hidden_states_predicted[-1] == stride_end_state:
            matches_ends = np.append(matches_ends, len(hidden_states_predicted))
        if hidden_states_predicted[0] == stride_start_state:
            matches_starts = np.concatenate([[0], matches_starts])

        # For each start, find the next end
        # if no end is found (as it is the end of the signal), we remove the start
        matches_start_end = []
        for start in matches_starts:
            with suppress(IndexError):
                matches_start_end.append([start, matches_ends[matches_ends > start][0]])

        # If multiple starts with the same end are found, we remove all but the first
        matches_start_end = np.array(matches_start_end)
        if len(matches_start_end) > 0:
            _, idx = np.unique(matches_start_end[:, 1], return_index=True)
            matches_start_end = matches_start_end[np.sort(idx)]

        return matches_start_end

    def _postprocess_matches(self, data, matches_start_end) -> np.ndarray:
        """Perform postprocessing step by snapping the stride border candidates to minima within the given data."""
        if self.snap_to_min_win_ms and len(matches_start_end) > 0:
            matches_start_end = snap_to_min(
                data[self.snap_to_min_axis].to_numpy(),
                matches_start_end,
                snap_to_min_win_samples=int(self.snap_to_min_win_ms / 1000 * self.sampling_rate_hz),
            )

        return matches_start_end
