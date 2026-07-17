"""Turn Roth HMM state predictions into gaitmap stride lists."""

from __future__ import annotations

from contextlib import suppress

import numpy as np
import pandas as pd
from tpcp import cf, make_action_safe
from typing_extensions import Self

from gaitmap.base import BaseStrideSegmentation
from gaitmap.stride_segmentation._utils import snap_to_min
from gaitmap.utils._algo_helper import invert_result_dictionary, set_params_from_dict
from gaitmap.utils._types import _Hashable
from gaitmap.utils.datatype_helper import SensorData, get_multi_sensor_names, is_sensor_data
from gaitmap_mad.stride_segmentation.hmm._composite_model import CompositeHmm
from gaitmap_mad.stride_segmentation.hmm._roth_model import RothSegmentationHmm


class HmmStrideSegmentation(BaseStrideSegmentation):
    """Segment regions from named HMM state groups.

    The wrapped model owns feature extraction and HMM inference. This class only
    converts the predicted stride-state sequence into stride borders and applies
    the optional signal-domain border refinement.

    Parameters
    ----------
    model
        Any fitted composite HMM. The bundled Roth model is used by default.
    segment_state_groups
        Named state groups to convert to segments. Selecting multiple groups
        adds their names as a ``type`` column to the output.
    snap_to_min_win_ms
        Search window for snapping predicted borders to local minima. Set to
        ``None`` to keep the borders derived directly from the state sequence.
    snap_to_min_axis
        Signal axis used for border refinement.

    Attributes
    ----------
    matches_start_end_
        Refined stride borders as an array or one array per sensor.
    stride_list_
        Refined stride borders in gaitmap stride-list format.
    matches_start_end_original_
        Stride borders derived directly from the hidden states.
    hidden_state_sequence_
        Predicted hidden states at the input sampling rate.
    result_model_
        Fitted model clone carrying the inference results, or one clone per sensor.

    Other Parameters
    ----------------
    data
        Single- or multi-sensor data passed to :meth:`segment`.
    sampling_rate_hz
        Sampling rate of the input data.

    """

    model: CompositeHmm
    segment_state_groups: tuple[str, ...]
    snap_to_min_win_ms: float | None
    snap_to_min_axis: str

    data: SensorData
    sampling_rate_hz: float

    matches_start_end_: pd.DataFrame | dict[str, pd.DataFrame]
    hidden_state_sequence_: np.ndarray | dict[str, np.ndarray]
    result_model_: CompositeHmm | dict[str, CompositeHmm]

    def __init__(
        self,
        model: CompositeHmm = cf(RothSegmentationHmm.from_pretrained()),
        *,
        segment_state_groups: tuple[str, ...] = ("stride",),
        snap_to_min_win_ms: float | None = 100,
        snap_to_min_axis: str = "gyr_ml",
    ) -> None:
        self.model = model
        self.segment_state_groups = segment_state_groups
        self.snap_to_min_win_ms = snap_to_min_win_ms
        self.snap_to_min_axis = snap_to_min_axis

    @property
    def stride_list_(self) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Return the detected stride borders in gaitmap format."""
        if isinstance(self.matches_start_end_, dict):
            return {sensor: self._format_stride_list(matches) for sensor, matches in self.matches_start_end_.items()}
        return self._format_stride_list(self.matches_start_end_)

    @property
    def matches_start_end_original_(self) -> pd.DataFrame | dict[_Hashable, pd.DataFrame]:
        """Return stride borders before signal-domain refinement."""
        if isinstance(self.hidden_state_sequence_, dict):
            return {
                sensor: self._hidden_states_to_matches_start_end(states)
                for sensor, states in self.hidden_state_sequence_.items()
            }
        return self._hidden_states_to_matches_start_end(self.hidden_state_sequence_)

    @make_action_safe
    def segment(self, data: SensorData, sampling_rate_hz: float, **_) -> Self:
        """Segment a single- or multi-sensor dataset."""
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        if is_sensor_data(data, check_gyr=False, check_acc=False) == "single":
            results = self._segment_single_dataset(data, sampling_rate_hz)
        else:
            results = invert_result_dictionary(
                {
                    sensor: self._segment_single_dataset(data[sensor], sampling_rate_hz)
                    for sensor in get_multi_sensor_names(data)
                }
            )
        set_params_from_dict(self, results, result_formatting=True)
        return self

    def _segment_single_dataset(self, data: pd.DataFrame, sampling_rate_hz: float) -> dict:
        model = self.model.clone().predict(data, sampling_rate_hz=sampling_rate_hz)
        matches = self._hidden_states_to_matches_start_end(model.hidden_state_sequence_)
        return {
            "matches_start_end": self._postprocess_matches(data, matches, sampling_rate_hz),
            "hidden_state_sequence": model.hidden_state_sequence_,
            "result_model_": model,
        }

    def _hidden_states_to_matches_start_end(self, states: np.ndarray) -> pd.DataFrame:
        if not self.segment_state_groups:
            raise ValueError("At least one segment state group must be selected.")
        matches = []
        typed = len(self.segment_state_groups) > 1
        for group in self.segment_state_groups:
            group_matches = self._group_states_to_matches(states, self.model.model.state_indices(group))
            frame = pd.DataFrame(group_matches, columns=["start", "end"])
            if typed:
                frame["type"] = group
            matches.append(frame)
        if not matches:
            return pd.DataFrame(columns=["start", "end"])
        return pd.concat(matches, ignore_index=True).sort_values(["start", "end"]).reset_index(drop=True)

    @staticmethod
    def _group_states_to_matches(states: np.ndarray, group_states: tuple[int, ...]) -> np.ndarray:
        if not group_states:
            raise ValueError("Segment state groups must contain at least one state.")
        stride_start_state, stride_end_state = group_states[0], group_states[-1]
        if len(states) == 0:
            return np.empty((0, 2), dtype=int)

        starts = np.flatnonzero(np.diff((states == stride_start_state).astype(np.int8)) > 0) + 1
        ends = np.flatnonzero(np.diff((states == stride_end_state).astype(np.int8)) < 0) + 1
        if states[0] == stride_start_state:
            starts = np.concatenate(([0], starts))
        if states[-1] == stride_end_state:
            ends = np.append(ends, len(states))

        matches = []
        for start in starts:
            with suppress(IndexError):
                matches.append((start, ends[ends > start][0]))

        if not matches:
            return np.empty((0, 2), dtype=int)
        matches_array = np.asarray(matches)
        _, unique_end_indices = np.unique(matches_array[:, 1], return_index=True)
        return matches_array[np.sort(unique_end_indices)]

    def _postprocess_matches(self, data: pd.DataFrame, matches: pd.DataFrame, sampling_rate_hz: float) -> pd.DataFrame:
        if self.snap_to_min_win_ms is None or len(matches) == 0:
            return matches
        result = matches.copy()
        result.loc[:, ["start", "end"]] = snap_to_min(
            data[self.snap_to_min_axis].to_numpy(),
            matches[["start", "end"]].to_numpy(),
            snap_to_min_win_samples=int(self.snap_to_min_win_ms / 1000 * sampling_rate_hz),
        )
        return result

    @staticmethod
    def _format_stride_list(matches: pd.DataFrame) -> pd.DataFrame:
        stride_list = matches.copy()
        stride_list.index.name = "s_id"
        return stride_list
