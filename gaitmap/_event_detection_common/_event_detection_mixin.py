"""Mixin for event detection algorithms that work similar to Rampp et al."""

import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Memory
from numpy.linalg import norm
from typing_extensions import Literal, Self

from gaitmap.utils._algo_helper import invert_result_dictionary, set_params_from_dict
from gaitmap.utils._types import _Hashable
from gaitmap.utils.array_handling import sliding_window_view
from gaitmap.utils.consts import BF_ACC, BF_GYR, SL_INDEX
from gaitmap.utils.datatype_helper import (
    SensorData,
    StrideList,
    get_multi_sensor_names,
    is_sensor_data,
    is_stride_list,
    set_correct_index,
)
from gaitmap.utils.exceptions import ValidationError
from gaitmap.utils.stride_list_conversion import (
    _stride_list_to_min_vel_single_sensor,
    enforce_stride_list_consistency,
)


class _EventDetectionMixin:
    memory: Optional[Memory]
    enforce_consistency: bool
    detect_only: Optional[Tuple[str, ...]]

    min_vel_event_list_: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]
    annotated_original_event_list_: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]

    data: SensorData
    sampling_rate_hz: float
    stride_list: pd.DataFrame
    input_stride_type: Literal["segmented", "ic"]

    def __init__(
        self,
        memory: Optional[Memory] = None,
        enforce_consistency: bool = True,
        detect_only: Optional[Tuple[str, ...]] = None,
        input_stride_type: Literal["segmented", "ic"] = "segmented",
    ):
        self.memory = memory
        self.enforce_consistency = enforce_consistency
        self.detect_only = detect_only
        self.input_stride_type = input_stride_type

    def detect(self, data: SensorData, stride_list: StrideList, *, sampling_rate_hz: float) -> Self:
        """Find gait events in data within strides provided by stride_list.

        Parameters
        ----------
        data
            The data set holding the imu raw data
        stride_list
            A list of strides provided by a stride segmentation method
        sampling_rate_hz
            The sampling rate of the data

        Returns
        -------
        self
            The class instance with all result attributes populated

        """
        dataset_type = is_sensor_data(data, frame="body")
        stride_list_type = is_stride_list(stride_list, stride_type="any")

        if dataset_type != stride_list_type:
            raise ValidationError(
                "An invalid combination of stride list and dataset was provided."
                f"The dataset is {dataset_type} sensor and the stride list is {stride_list_type} sensor."
            )

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.stride_list = stride_list

        detect_kwargs = self._get_detect_kwargs()

        if dataset_type == "single":
            results = self._detect_single_dataset(data, stride_list, detect_kwargs=detect_kwargs, memory=self.memory)
        else:
            results_dict: Dict[_Hashable, Dict[str, pd.DataFrame]] = {}
            for sensor in get_multi_sensor_names(data):
                results_dict[sensor] = self._detect_single_dataset(
                    data[sensor],
                    stride_list[sensor],
                    detect_kwargs=detect_kwargs,
                    memory=self.memory,
                )
            results = invert_result_dictionary(results_dict)

        if not self.enforce_consistency:
            results.pop("min_vel_event_list", None)
        set_params_from_dict(self, results, result_formatting=True)
        return self

    def _detect_single_dataset(
        self,
        data: pd.DataFrame,
        stride_list: pd.DataFrame,
        detect_kwargs: Dict[str, Any],
        memory: Memory,
    ) -> Dict[str, pd.DataFrame]:
        """Detect gait events for a single sensor data set and put into correct output stride list."""
        if memory is None:
            memory = Memory(None)

        acc = data[BF_ACC]
        gyr = data[BF_GYR]

        events = self.detect_only or ("ic", "tc", "min_vel")

        stride_list = set_correct_index(stride_list, SL_INDEX)

        # find events in all segments
        event_detection_func = self._select_all_event_detection_method()
        event_detection_func = memory.cache(event_detection_func)
        ic, tc, min_vel = event_detection_func(
            gyr, acc, stride_list, events=events, input_stride_type=self.input_stride_type, **detect_kwargs
        )

        # build first dict / df based on segment start and end
        annotated_original_event_list = {
            "s_id": stride_list.index,
            "start": stride_list["start"],
            "end": stride_list["end"],
        }
        for event, event_list in zip(("ic", "tc", "min_vel"), (ic, tc, min_vel)):
            if event in events:
                annotated_original_event_list[event] = event_list
        annotated_original_event_list = pd.DataFrame(annotated_original_event_list).set_index("s_id")
        if self.enforce_consistency:
            # check for consistency, remove inconsistent strides
            annotated_original_event_list, _ = enforce_stride_list_consistency(
                annotated_original_event_list, input_stride_type=self.input_stride_type, check_stride_list=False
            )

        if "min_vel" not in events or self.enforce_consistency is False:
            # do not set min_vel_event_list_ if consistency is not enforced as it would be completely scrambled
            # and can not be used for anything anyway
            return {"annotated_original_event_list": annotated_original_event_list}

        # convert to min_vel event list
        min_vel_event_list, _ = _stride_list_to_min_vel_single_sensor(
            annotated_original_event_list, source_stride_type=self.input_stride_type, target_stride_type="min_vel"
        )

        output_order = [c for c in ["start", "end", "ic", "tc", "min_vel", "pre_ic"] if c in min_vel_event_list.columns]

        # We enforce consistency again here, as a valid segmented stride list does not necessarily result in a valid
        # min_vel stride list
        min_vel_event_list, _ = enforce_stride_list_consistency(
            min_vel_event_list[output_order], input_stride_type="min_vel", check_stride_list=False
        )

        return {
            "min_vel_event_list": min_vel_event_list,
            "annotated_original_event_list": annotated_original_event_list,
        }

    def segmented_event_list_(self) -> Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
        warnings.deprecated(
            "`segmented_event_list_` is deprecated and will be removed in a future version. "
            "Use `annotated_original_event_list_` instead.",
        )
        return self.annotated_original_event_list_

    def _select_all_event_detection_method(self) -> Callable:
        """Select the function to calculate the all events.

        This is separate method to make it easy to overwrite by a subclass.
        """
        raise NotImplementedError()

    def _get_detect_kwargs(self) -> Dict[str, Any]:
        """Return a dictionary of keyword arguments that should be passed to the detect method.

        This is a separate method to make it easy to overwrite by a subclass.
        """
        return {}


def _detect_min_vel_gyr_energy(gyr: np.ndarray, min_vel_search_win_size: int) -> float:
    energy = norm(gyr, axis=-1) ** 2
    if min_vel_search_win_size >= len(energy):
        raise ValueError(
            f"min_vel_search_win_size_ms is {min_vel_search_win_size}, but gyr data has only {len(gyr)} samples. "
            "The search window should roughly be 100ms and the stride time must be larger. "
            "If the stride is shorter, something went wrong with stride segmentation."
        )
    energy = sliding_window_view(energy, window_length=min_vel_search_win_size, overlap=min_vel_search_win_size - 1)
    # find window with lowest summed energy
    min_vel_start = int(np.argmin(np.sum(energy, axis=1)))
    # min_vel event = middle of this window
    min_vel_center = min_vel_start + min_vel_search_win_size // 2
    return min_vel_center
