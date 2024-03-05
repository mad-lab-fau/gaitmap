"""An event detection algorithm optimized for stair ambulation developed by Liv Herzer in her Bachelor Thesis ."""
import warnings
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Memory
from scipy import signal
from tpcp import cf
from typing_extensions import Literal

from gaitmap._event_detection_common._event_detection_mixin import _detect_min_vel_gyr_energy, _EventDetectionMixin
from gaitmap.base import BaseEventDetection
from gaitmap.data_transform import BaseFilter, ButterworthFilter


class HerzerEventDetection(_EventDetectionMixin, BaseEventDetection):
    """Find gait events in the IMU raw signal based on signal characteristics.

    HerzerEventDetection uses signal processing approaches to find temporal gait events by searching for characteristic
    features in the foot-mounted sensor signals.
    The method was first developed in the context of the Bachelor Thesis by Liv Herzer (2021) under the supervision of
    Nils Roth [3]_.
    It combines techniques used in Rampp et al. (2014) [1]_ and Figueiredo et al. (2018) [2]_ with some original
    approaches.
    The particular goial was to create an Event Detection that worked well during level walking **and** stair
    ambulation.
    For more details refer to the `Notes` section.

    Parameters
    ----------
    min_vel_search_win_size_ms
        The size of the sliding window for finding the minimum gyroscope energy in ms.
    mid_swing_peak_prominence
        The expected min/(min, max) peak prominence of the mid swing peak in the `gyr_ml` signal.
        This value is passed directly to :func:`~scipy.signal.find_peaks`.
        The detected mid-swing peak is used to define the search region for the IC
    mid_swing_n_considered_peaks
        The number of peaks that should be considered in the initial search for the mid-swing.
        In particular for stair climbing, multiple prominent peaks might occure.
        The search algorithm consideres the n-most prominent and takes the one occuring first in the signal as the
        mid-swing.
        The detected mid-swing peak is used to define the search region for the IC
    ic_lowpass_filter
        An instance of a Filter-transform (e.g. :class:`~gaitmap.data_transform.ButterworthFilter`) that will be
        applied to the acc_pa signal before calculating the derivative when detecting the IC.
        While not enforced, this should be a lowpass filter to ensure that the results are as expected.
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

    >>> event_detection = HerzerEventDetection()
    >>> event_detection.detect(data=data, stride_list=stride_list, sampling_rate_hz=204.8)
    >>> event_detection.segmented_event_list_
          start    end       ic       tc  min_vel
    s_id
    0     48304  48558  48382.0  48304.0  48479.0
    1     48558  48788  48668.0  48558.0  48732.0
    2     48788  48994  48872.0  48788.0  48932.0
    3     48994  49201  49074.0  48994.0  49137.0
    4     49201  49422  49287.0  49201.0  49353.0
    ...     ...    ...      ...      ...      ...

    Notes
    -----
    This methods implements the detection of three events:

    terminal contact (`tc`), also called toe-off (TO):
        At `tc` the movement of the ankle joint changes from a plantar flexion to a dorsal extension in the sagittal
        plane.
        `tc` occurs roughly at the poit of the most rapid plantar flexion of the foot in the saggital plane.
        This corresponds to a minimum in the `gyr_ml` signal [2]_.

    initial contact (`ic`), also called heel strike (HS):
        At `ic` the foot decelerates rapidly when the foot hits the ground.
        However, looking simply for the point of highest deacceleration is not always robust.
        Therefore, this algorithm looks for the fastest change in acceleration.
        Specifically we are looking for the maximum in the derivative of the low-pass filtered acc_pa signal in the
        region after the swing phase.
        This usualy corresponds to a point closely after the minimum in the normal acc_pa signal.
        The apprach of looking at the derivative was inspired by [2]_.
        Note, that in the original bachelors thesis, the minimum of the derviative was searched.
        This is because the coordinate axis definition was different (corresponding to [1]_).
        To make sure that the correct peak is detected, as search window from the first prominent maximum of gyr_ml (
        swing phase) and 70% of the stride duration is defined.
        The search area is further narrowed down to be between the maximum of the low-pass filtered `acc_pa` signal and
        the steepest positive slope of the `gyr_ml` signal.
        After that the low-pass filtered `acc_pa` signal derivative is searched for a maximum in this area.

    minimal velocity (`min_vel_`), originally called mid stance (MS) in the paper [1]_:
        At `min_vel` the foot has the lowest velocity.
        It is defined to be the middle of the window with the lowest energy in all axes of the gyr signal.
        The default window size is set to 100 ms with 50 % overlap.
        The window size can be adjusted via the `min_vel_search_win_size_ms` parameter.
        This approach is identical to [1]_.

    The :func:`~gaitmap.event_detection.HerzerEventDetection.detect` method provides a stride list `min_vel_event_list`
    with the gait events mentioned above and additionally `start` and `end` of each stride, which are aligned to the
    `min_vel` samples.
    The start sample of each stride corresponds to the min_vel sample of that stride and the end sample corresponds to
    the min_vel sample of the subsequent stride.
    Furthermore, the `min_vel_event_list` list provides the `pre_ic` which is the ic event of the previous stride in
    the stride list.
    This function should NOT be used, since the detection of the min_vel gait events has not been validated
    for stair ambulation.
    Please refer to the `segmented_event_list_` instead.

    The :class:`~gaitmap.event_detection.HerzerEventDetection` includes a consistency check that is enabled by default.
    The gait events within one stride provided by the `stride_list` must occur in the expected order.
    Any stride where the gait events are detected in a different order or are not detected at all is dropped!
    For more infos on this see :func:`~gaitmap.utils.stride_list_conversion.enforce_stride_list_consistency`.
    If you wish to disable this consistency check, set `enforce_consistency` to False.
    In this case, the attribute `min_vel_event_list_` will not be set, but you can use `segmented_event_list_` to get
    all detected events for the exact stride list that was used as input.
    Note, that this list might contain NaN for some events.

    Furthermore, during the conversion from the segmented stride list to the "min_vel" stride list, breaks in
    continuous gait sequences (with continuous subsequent strides according to the `stride_list`) are detected and the
    first (segmented) stride of each sequence is dropped.
    This is required due to the shift of stride borders between the `stride_list` and the `min_vel_event_list`.
    Thus, the first segmented stride of a continuous sequence only provides a pre_ic and a min_vel sample for
    the first stride in the `min_vel_event_list`.
    Therefore, the `min_vel_event_list` list has one stride less per gait sequence than the `segmented_stride_list`.

    Further information regarding the coordinate system can be found :ref:`here<coordinate_systems>` and regarding the
    different types of strides can be found :ref:`here<stride_list_guide>`.

    Differences to original Implementation (Bachelorthesis by Liv Herzer [3]_):
        1. This implementation does not use an adaptive cut-off for the
           low-pass filter.
        2. Instead of iteratively lowering the detection threshold to find the maximum of the swing phase, we simply use
           `find_peaks` to search for the peak with the highest prominence that is still over some baseline threshold.
        3. The coordinate system that was considered in the BA was different to the gaitmap bodyframe.
           This means the acc_pa axis was upside down.


    .. [1] Rampp, A., Barth, J., Schülein, S., Gaßmann, K. G., Klucken, J., & Eskofier, B. M. (2014). Inertial
       sensor-based stride parameter calculation from gait sequences in geriatric patients. IEEE transactions on
       biomedical engineering, 62(4), 1089-1097.. https://doi.org/10.1109/TBME.2014.2368211
    .. [2] J. Figueiredo, P. Félix, L. Costa, J. C. Moreno and C. P. Santos, "Gait Event Detection in Controlled and
       Real-Life Situations: Repeated Measures From Healthy Subjects," in IEEE Transactions on Neural Systems and
       Rehabilitation Engineering, vol. 26, no. 10, pp. 1945-1956, Oct. 2018, doi: 10.1109/TNSRE.2018.2868094.
    .. [3] https://www.mad.tf.fau.de/person/liv-herzer-2/

    """

    min_vel_search_win_size_ms: float
    mid_swing_peak_prominence: Union[Tuple[float, float], float]
    mid_swing_n_considered_peaks: int
    ic_lowpass_filter: BaseFilter
    memory: Optional[Memory]
    enforce_consistency: bool
    stride_type: Literal["segmented"]

    def __init__(
        self,
        min_vel_search_win_size_ms: float = 100,
        mid_swing_peak_prominence: Union[Tuple[float, float], float] = 20,
        mid_swing_n_considered_peaks: int = 3,
        ic_lowpass_filter: BaseFilter = cf(ButterworthFilter(order=1, cutoff_freq_hz=4)),
        memory: Optional[Memory] = None,
        enforce_consistency: bool = True,
        detect_only: Optional[Tuple[str, ...]] = None,
            stride_type: Literal["segmented"] = "segmented",
    ):
        self.min_vel_search_win_size_ms = min_vel_search_win_size_ms
        self.mid_swing_peak_prominence = mid_swing_peak_prominence
        self.mid_swing_n_considered_peaks = mid_swing_n_considered_peaks
        self.ic_lowpass_filter = ic_lowpass_filter
        self.stride_type = stride_type
        super().__init__(memory=memory, enforce_consistency=enforce_consistency, detect_only=detect_only,
                         stride_type=stride_type)

    def _get_detect_kwargs(self) -> Dict[str, int]:
        min_vel_search_win_size = int(self.min_vel_search_win_size_ms / 1000 * self.sampling_rate_hz)
        return {
            "min_vel_search_win_size": min_vel_search_win_size,
            "mid_swing_peak_prominence": self.mid_swing_peak_prominence,
            "mid_swing_n_considered_peaks": self.mid_swing_n_considered_peaks,
            "sampling_rate_hz": self.sampling_rate_hz,
            "ic_lowpass_filter": self.ic_lowpass_filter,
        }

    def _select_all_event_detection_method(self) -> Callable:
        """Select the function to calculate the all events.

        This is separate method to make it easy to overwrite by a subclass.
        """
        return _find_all_events


def _find_all_events(
    gyr: pd.DataFrame,
    acc: pd.DataFrame,
    stride_list: pd.DataFrame,
    *,
    events: Tuple[str, ...] = ("ic", "tc", "min_vel"),
    min_vel_search_win_size: int,
    mid_swing_peak_prominence: Union[Tuple[float, float], float],
    mid_swing_n_considered_peaks: int,
    ic_lowpass_filter: BaseFilter,
    sampling_rate_hz: float,
    stride_type: Literal["segmented"]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Find events in provided data by looping over single strides."""
    warnings.warn("This algorithm works on "+stride_type+" stride type, IC stride type is not supoorted")
    gyr_ml = gyr["gyr_ml"].to_numpy()
    gyr = gyr.to_numpy()
    # inverting acc, as this algorithm was developed assuming a flipped axis like the original Rampp algorithm
    acc_pa = -acc["acc_pa"].to_numpy()
    ic_events = []
    tc_events = []
    min_vel_events = []
    for _, stride in stride_list.iterrows():
        start = stride["start"]
        end = stride["end"]
        gyr_sec = gyr[start:end]
        gyr_ml_sec = gyr_ml[start:end]
        duration = end - start
        # search for TC shortly before and after the labeled stride start
        # If for some reason, we are right at the beginning of a recording, we will start at 0
        tc_start = np.clip(int(start - 0.25 * duration), 0, None)
        gyr_ml_tc_sec = gyr_ml[tc_start : int(start + 0.25 * duration)]
        acc_sec = acc_pa[start:end]
        gyr_grad = np.gradient(gyr_ml_sec)

        if "ic" in events:
            ics = _detect_ic(
                gyr_ml_sec,
                acc_sec,
                gyr_grad,
                peak_prominence_thresholds=mid_swing_peak_prominence,
                n_considered_peaks=mid_swing_n_considered_peaks,
                lowpass_filter=ic_lowpass_filter,
                sampling_rate_hz=sampling_rate_hz,
            )

            ic_events.append(start + ics)
        if "tc" in events:
            tc_events.append(tc_start + _detect_tc(gyr_ml_tc_sec))
        if "min_vel" in events:
            min_vel_events.append(start + _detect_min_vel_gyr_energy(gyr_sec, min_vel_search_win_size))

    return (
        np.array(ic_events, dtype=float) if ic_events else None,
        np.array(tc_events, dtype=float) if tc_events else None,
        np.array(min_vel_events, dtype=float) if min_vel_events else None,
    )


def _get_midswing_max(gyr_ml, peak_prominence_thresholds: Union[Tuple[float, float], float], n_considered_peaks: int):
    """Return the first prominent maximum within the given stride.

    This maximum should correspond to the mid swing gait event.
    """
    peaks, properties = signal.find_peaks(gyr_ml, prominence=peak_prominence_thresholds, height=0)
    if len(peaks) == 0:
        return np.nan
    peak_prominence = properties["prominences"]
    # We find the n most prominent peaks and then take the one that occurs first in the data array.
    peak = peaks[np.min(np.argsort(peak_prominence)[-n_considered_peaks:])]

    return peak


def _detect_ic(
    gyr_ml: np.ndarray,
    acc_pa_inv: np.ndarray,
    gyr_ml_grad: np.ndarray,
    peak_prominence_thresholds: Union[Tuple[float, float], float],
    n_considered_peaks: int,
    lowpass_filter: BaseFilter,
    sampling_rate_hz: float,
) -> int:
    """Detect IC within the stride.

    Note, that this implementation expects the inverted signal of acc_pa compared to the normal bodyframe definition
    in gaitmap.
    This is because the algorithm was originally developed considering a different coordinate system.
    To keep the logic identical to the original paper, we pass in the inverted signal axis (see parent function)

    The IC is located at the minimum of the derivative of the low-pass filtered acc_pa_inv signal.
    The search for this minimum starts after the gyro_ml midswing peak and the filtered acc_pa_inv swing peak
    and ends at the pre-midstance peak in the gyro_ml signal or at 70% of the stride time.
    (This is better fit for various walking speeds and stair inclinations.)
    """
    # Determine rough search region
    # use midswing peak instead of global peak (the latter fails for stair descent)
    mid_swing_max = _get_midswing_max(gyr_ml, peak_prominence_thresholds, n_considered_peaks)
    search_region = (mid_swing_max, int(0.7 * len(gyr_ml)))

    if np.isnan(search_region[0]) or search_region[1] - search_region[0] <= 0:
        # The gyr argmax was not found in the first half of the step
        return np.nan

    # start with end as this max in the derivative is quite easily detectable within the search region
    # and the start acc_pa max has to be the max before that which is not necessarily
    # the global max within the search region
    refined_search_region_end = int(
        search_region[0]
        + np.argmax(gyr_ml_grad[slice(*search_region)])
        + 1
        # +1 because the min max distance is often very small
        # and in a search range the last value is normally not included but here it should be
    )
    refined_search_region_end = np.clip(refined_search_region_end, None, len(gyr_ml))

    # Low pass filter acc_pa to remove sharp peaks and get the overall shape of the signal.
    # The swing acceleration peak is now clearly distinguishable from the IC peak, because the latter is too high
    # frequency.
    acc_pa_inv_filt = lowpass_filter.filter(acc_pa_inv, sampling_rate_hz=sampling_rate_hz).filtered_data_
    try:
        # maximum of acc_pa signal after mid swing gyro peak, before gyro derivative max
        refined_search_region_start = int(
            search_region[0] + np.argmax(acc_pa_inv_filt[search_region[0] : refined_search_region_end])
        )
        refined_search_region_start = np.clip(refined_search_region_start, 0, None)
    except ValueError:
        return np.nan

    if refined_search_region_end - refined_search_region_start <= 0:
        return np.nan

    # the minimum in the derivative of the filtered acc_pa signal is our IC
    acc_pa_filt_deriv = np.diff(acc_pa_inv_filt)
    return refined_search_region_start + np.argmin(
        acc_pa_filt_deriv[refined_search_region_start:refined_search_region_end]
    )


def _detect_tc(gyr_ml: np.ndarray) -> float:
    """Detect TC in stride.

    The TC is located at the gyr_ml minimum somewhere around the given stride start.
    The search area for the TC is determined in the `_find_all_events` function and
    only the relevant slice of the gyro signal is then passed on to this function.
    """
    return float(np.argmin(gyr_ml))
