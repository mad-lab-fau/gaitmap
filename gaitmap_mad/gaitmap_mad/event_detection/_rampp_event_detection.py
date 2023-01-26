"""The event detection algorithm by Rampp et al. 2014."""
from typing import Callable, Dict, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from joblib import Memory

from gaitmap._event_detection_common._event_detection_mixin import _detect_min_vel_gyr_energy, _EventDetectionMixin
from gaitmap.base import BaseEventDetection
from gaitmap.data_transform import BaseFilter


class RamppEventDetection(_EventDetectionMixin, BaseEventDetection):
    """Find gait events in the IMU raw signal based on signal characteristics.

    RamppEventDetection uses signal processing approaches to find temporal gait events by searching for characteristic
    features in the foot-mounted sensor signals as described in Rampp et al. (2014) [1]_.
    For more details refer to the `Notes` section.

    Parameters
    ----------
    ic_search_region_ms
        The region to look for the initial contact in the acc_pa signal in ms given an ic candidate. According to [1]_,
        for the ic the algorithm first looks for a local minimum in the gyr_ml signal after the swing phase. The actual
        ic is then determined in the acc_pa signal in the ic_search_region_ms around that gyr_ml minimum.
        ic_search_region_ms[0] describes the start and ic_search_region_ms[1] the end of the region to check around the
        gyr_ml minimum. The values of `ic_search_region_ms` must be greater or equal than the sample time
        (1/`sampling_rate_hz`).
    min_vel_search_win_size_ms
        The size of the sliding window for finding the minimum gyroscope energy in ms.
    memory
        An optional `joblib.Memory` object that can be provided to cache the detection of all events.
    enforce_consistency
        An optional bool that can be set to False if you wish to disable postprocessing
        (see Notes section for more information).
    detect_only
        An optional tuple of strings that can be used to only detect a subset of events.
        By default, all events ("ic", "tc", "min_vel") are detected.
        If `min_vel` is not detected, the `min_vel_event_list_` output will not be available.
        If "ic" is not detected, the `pre_ic` will also not be available in the output.

    Attributes
    ----------
    min_vel_event_list_ : A stride list or dictionary with such values
        The result of the `detect` method holding all temporal gait events and start / end of all strides.
        The stride borders for the stride_events are aligned with the min_vel samples.
        Hence, the start sample of each stride corresponds to the min_vel sample of that stride and the end sample
        corresponds to the min_vel sample of the subsequent stride.
        Strides for which no valid events could be found are removed.
        Additional strides might have been removed due to the conversion from segmented to min_vel strides.
        The 's_id' index is selected according to which segmented stride the pre-ic belongs to.

    segmented_event_list_ : A stride list or dictionary with such values
        The result of the `detect` method holding all temporal gait events and start / end of all strides.
        This version of the results has the same stride borders than the input `stride_list` and has additional columns
        for all the detected events.
        Strides for which no valid events could be found are removed.


    Other Parameters
    ----------------
    data
        The data passed to the `detect` method.
    sampling_rate_hz
        The sampling rate of the data
    stride_list
        A list of strides provided by a stride segmentation method. The stride list is expected to have no gaps
        between subsequent strides. That means for subsequent strides the end sample of one stride should be the
        start sample of the next stride.

    Examples
    --------
    Get gait events from single sensor signal

    >>> event_detection = RamppEventDetection()
    >>> event_detection.detect(data=data, stride_list=stride_list, sampling_rate_hz=204.8)
    >>> event_detection.min_vel_event_list_
           start     end      ic      tc  min_vel  pre_ic
    s_id
    0      519.0   710.0   651.0   584.0    519.0   498.0
    1      710.0   935.0   839.0   802.0    710.0   651.0
    2      935.0  1183.0  1089.0  1023.0    935.0   839.0
    ...

    Notes
    -----
    Rampp et al. implemented the detection of three gait events from foot-mounted sensor data:

    terminal contact (`tc`), originally called toe-off (TO) in the paper [1]_:
        At `tc` the movement of the ankle joint changes from a plantar flexion to a dorsal extension in the sagittal
        plane.
        This change results in a zero crossing of the gyr_ml signal.
        Also refer to the :ref:`image below <fe>`.

    initial contact (`ic`), originally called heel strike (HS) in the paper [1]_:
        At `ic` the foot decelerates rapidly when the foot hits the ground.
        For the detection of `ic` only the signal between the absolute maximum and the end of the first half of the
        gyr_ml signal is considered.
        Within this segment, `ic` is found by searching for the minimum between the point of the steepest negative
        slope and the point of the steepest positive slope in the following signal.
        After that the acc_pa signal is searched for a maximum in the area before and after the described minimum in
        the gyr_ml signal.
        In the original implementation of the paper, this was actually a minimum due to flipped sensor coordinate axes.
        The default search window is set to 80 ms before and 50 ms after the minimum.
        The search window borders can be adjusted via the `ic_search_region_ms` parameter.
        Also refer to the :ref:`image below <fe>`.

    minimal velocity (`min_vel_`), originally called mid stance (MS) in the paper [1]_:
        At `min_vel` the foot has the lowest velocity.
        It is defined to be the middle of the window with the lowest energy in all axes of the gyr signal.
        The default window size is set to 100 ms with 50 % overlap.
        The window size can be adjusted via the `min_vel_search_win_size_ms` parameter.
        Also refer to the :ref:`image below <fe>`.

    The :func:`~gaitmap.event_detection.RamppEventDetection.detect` method provides a stride list `min_vel_event_list`
    with the gait events mentioned above and additionally `start` and `end` of each stride, which are aligned to the
    `min_vel` samples.
    The start sample of each stride corresponds to the min_vel sample of that stride and the end sample corresponds to
    the min_vel sample of the subsequent stride.
    Furthermore, the `min_vel_event_list` list provides the `pre_ic` which is the ic event of the previous stride in
    the stride list.

    The :class:`~gaitmap.event_detection.RamppEventDetection` includes a consistency check that is enabled by default.
    The gait events within one stride provided by the `stride_list` must occur in the expected order.
    Any stride where the gait events are detected in a different order or are not detected at all is dropped!
    For more infos on this see :func:`~gaitmap.utils.stride_list_conversion.enforce_stride_list_consistency`.
    If you wish to disable this consistency check, set `enforce_consistency` to False.
    In this case, the attribute `min_vel_event_list_` will not be set, but you can use `segmented_event_list_` to get
    all detected events for the exact stride list that was used as input.
    Note, that this list might contain NaN for some events.

    Furthermore, during the conversion from the segmented stride list to the "min_vel" stride list, breaks in
    continuous gait sequences ( with continuous subsequent strides according to the `stride_list`) are detected and the
    first (segmented) stride of each sequence is dropped.
    This is required due to the shift of stride borders between the `stride_list` and the `min_vel_event_list`.
    Thus, the first segmented stride of a continuous sequence only provides a pre_ic and a min_vel sample for
    the first stride in the `min_vel_event_list`.
    Therefore, the `min_vel_event_list` list has one stride less per gait sequence than the `segmented_stride_list`.

    Further information regarding the coordinate system can be found :ref:`here<coordinate_systems>` and regarding the
    different types of strides can be found :ref:`here<stride_list_guide>`.

    The image below gives an overview about the events and where they occur in the signal.

    .. _fe:
    .. figure:: /images/event_detection.svg

    .. [1] Rampp, A., Barth, J., Schülein, S., Gaßmann, K. G., Klucken, J., & Eskofier, B. M. (2014). Inertial
       sensor-based stride parameter calculation from gait sequences in geriatric patients. IEEE transactions on
       biomedical engineering, 62(4), 1089-1097.. https://doi.org/10.1109/TBME.2014.2368211

    """

    ic_search_region_ms: Tuple[float, float]
    min_vel_search_win_size_ms: float

    def __init__(
        self,
        ic_search_region_ms: Tuple[float, float] = (80, 50),
        min_vel_search_win_size_ms: float = 100,
        memory: Optional[Memory] = None,
        enforce_consistency: bool = True,
        detect_only: Optional[Tuple[str, ...]] = None,
    ):
        self.ic_search_region_ms = ic_search_region_ms
        self.min_vel_search_win_size_ms = min_vel_search_win_size_ms
        super().__init__(memory=memory, enforce_consistency=enforce_consistency, detect_only=detect_only)

    def _select_all_event_detection_method(self) -> Callable:
        """Select the function to calculate the all events.

        This is separate method to make it easy to overwrite by a subclass.
        """
        return _find_all_events

    def _get_detect_kwargs(self) -> Dict[str, Union[Tuple[int, int], int]]:
        ic_search_region = cast(
            Tuple[int, int], tuple(int(v / 1000 * self.sampling_rate_hz) for v in self.ic_search_region_ms)
        )
        if all(v == 0 for v in ic_search_region):
            raise ValueError(
                "The chosen values are smaller than the sample time ({} ms)".format((1 / self.sampling_rate_hz) * 1000)
            )
        min_vel_search_win_size = int(self.min_vel_search_win_size_ms / 1000 * self.sampling_rate_hz)
        return {
            "ic_search_region": ic_search_region,
            "min_vel_search_win_size": min_vel_search_win_size,
            "sampling_rate_hz": self.sampling_rate_hz,
            "gyr_ic_lowpass_filter": None,
        }


def _find_all_events(
    gyr: pd.DataFrame,
    acc: pd.DataFrame,
    stride_list: pd.DataFrame,
    events: Tuple[str, ...],
    ic_search_region: Tuple[float, float],
    min_vel_search_win_size: int,
    sampling_rate_hz: float,
    gyr_ic_lowpass_filter: Optional[BaseFilter],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find events in provided data by looping over single strides."""
    gyr_ml = gyr["gyr_ml"]

    if "ic" in events:
        if gyr_ic_lowpass_filter is not None:
            gyr_ml_filtered = gyr_ic_lowpass_filter.filter(
                gyr_ml, sampling_rate_hz=sampling_rate_hz
            ).filtered_data_.to_numpy()
        else:
            gyr_ml_filtered = gyr_ml.to_numpy()
    else:
        gyr_ml_filtered = None
    gyr = gyr.to_numpy()
    gyr_ml = gyr_ml.to_numpy()
    acc_pa = -acc["acc_pa"].to_numpy()  # have to invert acc data to work on rampp paper
    ic_events = []
    tc_events = []
    min_vel_events = []
    for _, stride in stride_list.iterrows():
        start = stride["start"]
        end = stride["end"]
        if "ic" in events:
            gyr_ml_filtered_sec = gyr_ml_filtered[start:end]
            acc_sec = acc_pa[start:end]
            gyr_grad = np.gradient(gyr_ml[start:end])
            ic_events.append(start + _detect_ic(gyr_ml_filtered_sec, acc_sec, gyr_grad, ic_search_region))
        if "tc" in events:
            tc_events.append(start + _detect_tc(gyr_ml[start:end]))
        if "min_vel" in events:
            min_vel_events.append(start + _detect_min_vel_gyr_energy(gyr[start:end], min_vel_search_win_size))

    return (
        np.array(ic_events, dtype=float) if ic_events else None,
        np.array(tc_events, dtype=float) if tc_events else None,
        np.array(min_vel_events, dtype=float) if min_vel_events else None,
    )


def _detect_ic(
    gyr_ml: np.ndarray, acc_pa_inv: np.ndarray, gyr_ml_grad: np.ndarray, ic_search_region: Tuple[float, float]
) -> float:
    """Find the ic.

    Note, that this implementation expects the inverted signal of acc_pa compared to the normal bodyframe definition
    in gaitmap.
    This is because the algorithm was originally developed considering a different coordinate system.
    To keep the logic identical to the original paper, we pass in the inverted signal axis (see parent function)
    """
    # Determine rough search region
    search_region = (np.argmax(gyr_ml), int(0.6 * gyr_ml.shape[0]))

    if search_region[1] - search_region[0] <= 0:
        # The gyr argmax was not found in the first half of the step
        return np.nan

    # alternative:
    # refined_search_region_start, refined_search_region_end = search_region
    refined_search_region_start = int(search_region[0] + np.argmin(gyr_ml_grad[slice(*search_region)]))
    refined_search_region_end = int(
        refined_search_region_start + np.argmax(gyr_ml_grad[refined_search_region_start : search_region[1]])
    )

    if refined_search_region_end - refined_search_region_start <= 0:
        return np.nan

    # Find heel strike candidate in search region based on gyr
    heel_strike_candidate = refined_search_region_start + np.argmin(
        gyr_ml[refined_search_region_start:refined_search_region_end]
    )

    # Acc search window
    acc_search_region_start = int(np.max(np.array([0, heel_strike_candidate - ic_search_region[0]])))
    acc_search_region_end = int(np.min(np.array([gyr_ml.shape[0], heel_strike_candidate + ic_search_region[1]])))

    return float(acc_search_region_start + np.argmin(acc_pa_inv[acc_search_region_start:acc_search_region_end]))


def _detect_tc(gyr_ml: np.ndarray) -> float:
    try:
        return np.where(np.diff(np.signbit(gyr_ml)))[0][0]
    except IndexError:
        return np.nan
