"""The event detection algorithm by Rampp et al. 2014 slightly changed to better fit stair ambulation."""
from typing import Optional, Tuple, Union, Dict, TypeVar, Callable

import numpy as np
import pandas as pd
from joblib import Memory
from numpy.linalg import norm
from scipy import signal

from gaitmap.base import BaseEventDetection
from gaitmap.utils._algo_helper import invert_result_dictionary, set_params_from_dict
from gaitmap.utils._types import _Hashable
from gaitmap.utils.array_handling import sliding_window_view
from gaitmap.utils.consts import BF_ACC, BF_GYR, SL_INDEX
from gaitmap.utils.datatype_helper import (
    StrideList,
    SensorData,
    get_multi_sensor_names,
    set_correct_index,
    is_sensor_data,
    is_stride_list,
)
from gaitmap.utils.exceptions import ValidationError
from gaitmap.utils.stride_list_conversion import (
    enforce_stride_list_consistency,
    _segmented_stride_list_to_min_vel_single_sensor,
)

Self = TypeVar("Self", bound="ComboEventDetection")


class ComboEventDetection(BaseEventDetection):
    """Find gait events in the IMU raw signal based on signal characteristics.

    ComboEventDetection uses signal processing approaches to find temporal gait events by searching for characteristic
    features in the foot-mounted sensor signals.
    For more details refer to the `Notes` section.

    Parameters
    ----------
    min_vel_search_win_size_ms
        The size of the sliding window for finding the minimum gyroscope energy in ms.
    memory
        An optional `joblib.Memory` object that can be provided to cache the detection of all events.
    enforce_consistency
        An optional bool that can be set to False if you wish to disable postprocessing
        (see Notes section for more information).

    Attributes
    ----------
    min_vel_event_list_ : A stride list or dictionary with such values !DO NOT USE!
        The result of the `detect` method holding all temporal gait events and start / end of all strides.
        The stride borders for the stride_events are aligned with the min_vel samples.
        Hence, the start sample of each stride corresponds to the min_vel sample of that stride and the end sample
        corresponds to the min_vel sample of the subsequent stride.
        Strides for which no valid events could be found are removed.
        Additional strides might have been removed due to the conversion from segmented to min_vel strides.
        DO NOT USE this attribute: `ic` and `tc` detection have been thorouhly tested, however detected min_vel events
        have NOT been validated!
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

    >>> event_detection = ComboEventDetection()
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
    Rampp et al. implemented the detection of three gait events from foot-mounted sensor data [1]_.
    This detection was improved for stair ambulation with the following detection rules:

    terminal contact (`tc`), also called toe-off (TO):
        `tc` occurs roughly at the poit of the most rapid plantar flexion of the foot in the saggital plane.
        This results in a minimum in the gyr_ml signal.

    initial contact (`ic`), also called heel strike (HS):
        At `ic` the foot decelerates rapidly when the foot hits the ground.
        For the detection of `ic` only the signal between the first prominent maximum and 70% of the
        gyr_ml signal is considered.
        The search area is further narrowed down to in between the maximum of the low-pass filtered acc_pa signal and
        the steepest positive slope of the gyr_ml signal.
        After that the low-pass filtered acc_pa signal derivative is searched for a minimum in this area.

    minimal velocity (`min_vel_`), originally called mid stance (MS) in the paper [1]_:
        At `min_vel` the foot has the lowest velocity.
        It is defined to be the middle of the window with the lowest energy in all axes of the gyr signal.
        The default window size is set to 100 ms with 50 % overlap.
        The window size can be adjusted via the `min_vel_search_win_size_ms` parameter.

    The :func:`~gaitmap.event_detection.ComboEventDetection.detect` method provides a stride list `min_vel_event_list`
    with the gait events mentioned above and additionally `start` and `end` of each stride, which are aligned to the
    `min_vel` samples.
    The start sample of each stride corresponds to the min_vel sample of that stride and the end sample corresponds to
    the min_vel sample of the subsequent stride.
    Furthermore, the `min_vel_event_list` list provides the `pre_ic` which is the ic event of the previous stride in
    the stride list.
    This function should NOT be used, since the detection of the min_vel gait events has not been validated
    for stair ambulation.
    Please refer to the `segmented_event_list_` instead.

    The :class:`~gaitmap.event_detection.ComboEventDetection` includes a consistency check that is enabled by default.
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


    .. [1] Rampp, A., Barth, J., Schülein, S., Gaßmann, K. G., Klucken, J., & Eskofier, B. M. (2014). Inertial
       sensor-based stride parameter calculation from gait sequences in geriatric patients. IEEE transactions on
       biomedical engineering, 62(4), 1089-1097.. https://doi.org/10.1109/TBME.2014.2368211

    """

    min_vel_search_win_size_ms: float
    memory: Optional[Memory]
    enforce_consistency: bool

    min_vel_event_list_: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]
    segmented_event_list_: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]

    data: SensorData
    sampling_rate_hz: float
    stride_list: pd.DataFrame

    def __init__(
        self,
        min_vel_search_win_size_ms: float = 100,
        memory: Optional[Memory] = None,
        enforce_consistency: bool = True,
    ):
        self.min_vel_search_win_size_ms = min_vel_search_win_size_ms
        self.memory = memory
        self.enforce_consistency = enforce_consistency

    def detect(self: Self, data: SensorData, stride_list: StrideList, sampling_rate_hz: float) -> Self:
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
                "The dataset is {} sensor and the stride list is {} sensor.".format(dataset_type, stride_list_type)
            )

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.stride_list = stride_list

        min_vel_search_win_size = int(self.min_vel_search_win_size_ms / 1000 * self.sampling_rate_hz)

        if dataset_type == "single":
            results = self._detect_single_dataset(
                data,
                stride_list,
                min_vel_search_win_size,
                memory=self.memory,
            )
        else:
            results_dict: Dict[_Hashable, Dict[str, pd.DataFrame]] = dict()
            for sensor in get_multi_sensor_names(data):
                results_dict[sensor] = self._detect_single_dataset(
                    data[sensor],
                    stride_list[sensor],
                    min_vel_search_win_size,
                    memory=self.memory,
                )
            results = invert_result_dictionary(results_dict)

        # do not set min_vel_event_list_ if consistency is not enforced as it would be completely scrambeled
        # and can not be used for anything anyway
        if not self.enforce_consistency:
            del results["min_vel_event_list"]
        set_params_from_dict(self, results, result_formatting=True)
        return self

    def _detect_single_dataset(
        self,
        data: pd.DataFrame,
        stride_list: pd.DataFrame,
        min_vel_search_win_size: int,
        memory: Memory,
    ) -> Dict[str, pd.DataFrame]:
        """Detect gait events for a single sensor data set and put into correct output stride list."""
        if memory is None:
            memory = Memory(None)

        acc = data[BF_ACC]
        gyr = data[BF_GYR]

        stride_list = set_correct_index(stride_list, SL_INDEX)

        # find events in all segments
        event_detection_func = self._select_all_event_detection_method()
        event_detection_func = memory.cache(event_detection_func)
        ic, tc, min_vel = event_detection_func(gyr, acc, stride_list, min_vel_search_win_size)

        # build first dict / df based on segment start and end
        segmented_event_list = {
            "s_id": stride_list.index,
            "start": stride_list["start"],
            "end": stride_list["end"],
            "ic": ic,
            "tc": tc,
            "min_vel": min_vel,
        }
        segmented_event_list = pd.DataFrame(segmented_event_list).set_index("s_id")

        if self.enforce_consistency:
            # check for consistency, remove inconsistent strides
            segmented_event_list, _ = enforce_stride_list_consistency(
                segmented_event_list, stride_type="segmented", check_stride_list=False
            )

        min_vel_event_list, _ = _segmented_stride_list_to_min_vel_single_sensor(
            segmented_event_list, target_stride_type="min_vel"
        )

        min_vel_event_list = min_vel_event_list[["start", "end", "ic", "tc", "min_vel", "pre_ic"]]

        return {
            "min_vel_event_list": min_vel_event_list,
            "segmented_event_list": segmented_event_list,
        }

    def _select_all_event_detection_method(self) -> Callable:  # noqa: no-self-use
        """Select the function to calculate the all events.

        This is separate method to make it easy to overwrite by a subclass.
        """
        return _find_all_events


def _find_all_events(
    gyr: pd.DataFrame,
    acc: pd.DataFrame,
    stride_list: pd.DataFrame,
    min_vel_search_win_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find events in provided data by looping over single strides."""
    gyr_ml = gyr["gyr_ml"].to_numpy()
    gyr = gyr.to_numpy()
    acc_pa = -acc["acc_pa"].to_numpy()  # have to invert acc data to work on rampp paper
    ic_events = []
    fc_events = []
    min_vel_events = []
    for _, stride in stride_list.iterrows():
        start = stride["start"]
        end = stride["end"]
        gyr_sec = gyr[start:end]
        gyr_ml_sec = gyr_ml[start:end]
        duration = end - start
        # search for TC shortly before and after the labeled stride start
        tc_start = int(start - 0.25 * duration)
        gyr_ml_tc_sec = gyr_ml[tc_start : int(start + 0.25 * duration)]
        acc_sec = acc_pa[start:end]
        gyr_grad = np.gradient(gyr_ml_sec)

        ic_events.append(start + _detect_ic(gyr_ml_sec, acc_sec, gyr_grad))
        fc_events.append(tc_start + _detect_tc(gyr_ml_tc_sec))
        min_vel_events.append(start + _detect_min_vel(gyr_sec, min_vel_search_win_size))

    return (
        np.array(ic_events, dtype=float),
        np.array(fc_events, dtype=float),
        np.array(min_vel_events, dtype=float),
    )


def _detect_min_vel(gyr: np.ndarray, min_vel_search_win_size: int) -> float:
    energy = norm(gyr, axis=-1) ** 2
    if min_vel_search_win_size >= len(energy):
        raise ValueError("The value chosen for min_vel_search_win_size_ms is too large. Should be 100 ms.")
    energy = sliding_window_view(
        energy,
        window_length=min_vel_search_win_size,
        overlap=min_vel_search_win_size - 1,
    )
    # find window with lowest summed energy
    min_vel_start = int(np.argmin(np.sum(energy, axis=1)))
    # min_vel event = middle of this window
    min_vel_center = min_vel_start + min_vel_search_win_size // 2
    return min_vel_center


def _get_midswing_max(gyr_ml):
    """ Return the first prominent maximum within the given stride.

    This maximum should correspond to the mid swing gait event.
    """
    peaks = []
    prominence = 800  # TODO: make this adjustable?

    # lower prominence condition until several peaks are recognized, to not only get THE prominent peak
    # (the mid swing peak is mostly the first out of several prominent peaks, but sometimes not the most prominent one)
    while len(peaks) < 2:
        if prominence <= 0:
            return np.nan

        # find positive peaks with the given prominence
        peaks, _ = signal.find_peaks(gyr_ml, prominence=prominence, height=0)

        if prominence > 20:
            prominence -= 20
        # some level walking steps only have one maximum over 0, this seems to be one of these steps
        elif len(peaks) == 1:
            break

    # only the first peak is interesting
    peak = peaks[0]
    return peak


def _detect_ic(
    gyr_ml: np.ndarray,
    acc_pa: np.ndarray,
    gyr_ml_grad: np.ndarray,
) -> float:
    """ Detect IC within the stride.

    The IC is located at the minimum of the derivative of the low-pass filtered acc_pa signal.
    The search for this minimum starts after the gyro_ml midswing peak and the filtered acc_pa swing peak
    and ends at the pre-midstance peak in the gyro_ml signal or at the latest at 70% of the stride length.
    (This is better fit for various walking speeds and stair inclinations.)
    """
    # Determine rough search region
    # use midswing peak instead of global peak (the latter fails for stair descent)
    mid_swing_max = _get_midswing_max(gyr_ml)
    search_region = (mid_swing_max, int(0.7 * len(gyr_ml)))

    # search_region = (np.argmax(gyr_ml), int(0.6 * len(gyr_ml)))

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

    # Low pass filter acc_pa to remove sharp peaks and get the overall shape of the signal.The swing acceleration peak
    # is now clearly distinguishable from the IC peak, because the latter is too high frequency.
    # First, design the Butterworth filter
    wn = 4 * 2 / len(gyr_ml)  # cutoff frequency at 4 * the stride frequency
    n = 1  # Filter order
    b, a = signal.butter(n, wn, output="ba")

    # Second, apply the filter
    acc_pa_filt = signal.filtfilt(b, a, acc_pa)
    try:
        # maximum of acc_pa signal after mid swing gyro peak, before gyro derivative max
        refined_search_region_start = int(
            search_region[0] + np.argmax(acc_pa_filt[search_region[0] : refined_search_region_end])
        )
    except ValueError:
        return np.nan

    if refined_search_region_end - refined_search_region_start <= 0:
        return np.nan

    # Acc search window
    acc_search_region_start = int(np.max(np.array([0, search_region[0], refined_search_region_start])))
    acc_search_region_end = int(
        np.min(np.array([len(acc_pa), refined_search_region_end]))
    )  # should always be the latter

    # the minimum in the derivative of the filtered acc_pa signal is our IC
    acc_pa_filt_deriv = np.diff(acc_pa_filt)
    return float(acc_search_region_start + np.argmin(acc_pa_filt_deriv[acc_search_region_start:acc_search_region_end]))


def _detect_tc(gyr_ml: np.ndarray) -> float:
    """ Detect TC in stride.

    The TC is located at the gyr_ml minimum somewhere around the given stride start.
    The search area for the TC is determined in the `_find_all_events` function and
    only the relevant slice of the gyro signal is then passed on to this function.
    """
    try:
        return np.argmin(gyr_ml)
    except IndexError:
        return np.nan
