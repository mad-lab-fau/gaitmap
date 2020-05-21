"""The event detection algorithm by Rampp et al. 2014."""
from typing import Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd
from numpy.linalg import norm

from gaitmap.base import BaseEventDetection, BaseType
from gaitmap.utils.stride_list_conversion import (
    enforce_stride_list_consistency,
    _segmented_stride_list_to_min_vel_single_sensor,
)
from gaitmap.utils.array_handling import sliding_window_view
from gaitmap.utils.consts import BF_ACC, BF_GYR
from gaitmap.utils.dataset_helper import (
    is_multi_sensor_dataset,
    is_single_sensor_dataset,
    is_single_sensor_stride_list,
    is_multi_sensor_stride_list,
    StrideList,
    Dataset,
    get_multi_sensor_dataset_names,
)


class RamppEventDetection(BaseEventDetection):
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
        gyr_ml minimum. The values of ic_search_region_ms must be greater or equal than the length of one sample.
    min_vel_search_win_size_ms
        The size of the sliding window for finding the minimum gyroscope energy in ms.

    Attributes
    ----------
    stride_events_ : A stride list or dictionary with such values
        The result of the `detect` method holding all temporal gait events and start / end of all strides. Formatted
        as pandas DataFrame. The stride borders for the stride_events are aligned with the min_vel samples. Hence,
        the start sample of each stride corresponds to the min_vel sample of that stride and the end sample corresponds
        to the min_vel sample of the subsequent stride.

    Other Parameters
    ----------------
    data
        The data passed to the `detect` method.
    sampling_rate_hz
        The sampling rate of the data
    segmented_stride_list
        A list of strides provided by a stride segmentation method. The stride list is expected to have no gaps
        between subsequent strides. That means for subsequent strides the end sample of one stride should be the
        start sample of the next stride.

    Examples
    --------
    Get gait events from single sensor signal

    >>> event_detection = RamppEventDetection()
    >>> event_detection.detect(data=data, sampling_rate_hz=204.8, segmented_stride_list=stride_list)
    >>> event_detection.stride_events_
        s_id   start     end      ic      tc  min_vel  pre_ic
    0      0   519.0   710.0   651.0   584.0    519.0   498.0
    1      1   710.0   935.0   839.0   802.0    710.0   651.0
    2      2   935.0  1183.0  1089.0  1023.0    935.0   839.0
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

    The :func:`~gaitmap.event_detection.RamppEventDetection.detect` method provides a stride list `stride_events_` with
    the gait events mentioned above and additionally `start` and `end` of each stride, which are aligned to the
    `min_vel` samples.
    The start sample of each stride corresponds to the min_vel sample of that stride and the end sample corresponds to
    the min_vel sample of the subsequent stride.
    Furthermore, the `stride_events_` list provides the `pre_ic` which is the ic event of the previous stride in the
    stride list.

    The :class:`~gaitmap.event_detection.RamppEventDetection` includes a consistency check.
    The gait events within one stride provided by the `segmented_stride_list` must occur in the order tc - ic - men_vel.
    Any stride where the gait events are detected in a different order is dropped!

    Furthermore, breaks in continuous gait sequences (with continuous subsequent strides according to the
    `segmented_stride_list`) are detected and the first (segmented) stride of each sequence is dropped.
    This is required due to the shift of stride borders between the `segmented_stride_list` and the `stride_events_`.
    Thus, the dropped first segmented_stride of a continuous sequence only provides a pre_ic and a min_vel sample for
    the first stride in the `stride_events_`. Therefore, the `stride_events_` list has one stride less than the
    `segmented_stride_list`.

    Further information regarding the coordinate system can be found :ref:`here<coordinate_systems>`.

    The image below gives an overview about the events and where they occur in the signal.

    .. _fe:
    .. figure:: /images/event_detection.svg

    .. [1] Rampp, A., Barth, J., Schülein, S., Gaßmann, K. G., Klucken, J., & Eskofier, B. M. (2014). Inertial
       sensor-based stride parameter calculation from gait sequences in geriatric patients. IEEE transactions on
       biomedical engineering, 62(4), 1089-1097.. https://doi.org/10.1109/TBME.2014.2368211

    """

    ic_search_region_ms: Tuple[float, float]
    min_vel_search_win_size_ms: float

    stride_events_: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]

    data: Dataset
    sampling_rate_hz: float
    segmented_stride_list: pd.DataFrame

    def __init__(self, ic_search_region_ms: Tuple[float, float] = (80, 50), min_vel_search_win_size_ms: float = 100):
        self.ic_search_region_ms = ic_search_region_ms
        self.min_vel_search_win_size_ms = min_vel_search_win_size_ms

    def detect(self: BaseType, data: Dataset, sampling_rate_hz: float, segmented_stride_list: StrideList) -> BaseType:
        """Find gait events in data within strides provided by segmented_stride_list.

        Parameters
        ----------
        data
            The data set holding the imu raw data
        sampling_rate_hz
            The sampling rate of the data
        segmented_stride_list
            A list of strides provided by a stride segmentation method

        Returns
        -------
        self
            The class instance with all result attributes populated

        """
        if is_single_sensor_dataset(data) and not is_single_sensor_stride_list(segmented_stride_list):
            raise ValueError("Provided stride list does not fit to provided single sensor data set")

        if is_multi_sensor_dataset(data) and not is_multi_sensor_stride_list(segmented_stride_list):
            raise ValueError("Provided stride list does not fit to provided multi sensor data set")

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.segmented_stride_list = segmented_stride_list

        ic_search_region = tuple(int(v / 1000 * self.sampling_rate_hz) for v in self.ic_search_region_ms)
        if all(v == 0 for v in ic_search_region):
            raise ValueError(
                "The chosen values are smaller than the sample time ({} ms)".format((1 / self.sampling_rate_hz) * 1000)
            )
        min_vel_search_win_size = int(self.min_vel_search_win_size_ms / 1000 * self.sampling_rate_hz)

        if is_single_sensor_dataset(data):
            self.stride_events_ = self._detect_single_dataset(
                data, segmented_stride_list, ic_search_region, min_vel_search_win_size
            )
        elif is_multi_sensor_dataset(data):
            self.stride_events_ = dict()
            for sensor in get_multi_sensor_dataset_names(data):
                self.stride_events_[sensor] = self._detect_single_dataset(
                    data[sensor], segmented_stride_list[sensor], ic_search_region, min_vel_search_win_size
                )
        else:
            raise ValueError("Provided data set is not supported by gaitmap")

        return self

    def _detect_single_dataset(
        self,
        data: pd.DataFrame,
        segmented_stride_list: pd.DataFrame,
        ic_search_region: Tuple[int, int],
        min_vel_search_win_size: int,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Detect gait events for a single sensor data set and put into correct output stride list."""
        acc = data[BF_ACC]
        gyr = data[BF_GYR]

        # find events in all segments
        ic, tc, min_vel = self._find_all_events(
            gyr, acc, segmented_stride_list, ic_search_region, min_vel_search_win_size
        )

        # build first dict / df based on segment start and end
        tmp_stride_event_dict = {
            "s_id": segmented_stride_list["s_id"],
            "start": segmented_stride_list["start"],
            "end": segmented_stride_list["end"],
            "ic": ic,
            "tc": tc,
            "min_vel": min_vel,
        }
        tmp_stride_event_df = pd.DataFrame(tmp_stride_event_dict)

        # check for consistency, remove inconsistent lines
        tmp_stride_event_df, _ = enforce_stride_list_consistency(
            tmp_stride_event_df, stride_type="segmented", check_stride_list=False
        )

        tmp_stride_event_df, _ = _segmented_stride_list_to_min_vel_single_sensor(
            tmp_stride_event_df, target_stride_type="min_vel"
        )

        stride_events_ = tmp_stride_event_df[["s_id", "start", "end", "ic", "tc", "min_vel", "pre_ic"]]

        return stride_events_

    @staticmethod
    def _find_all_events(
        gyr: pd.DataFrame,
        acc: pd.DataFrame,
        stride_list: pd.DataFrame,
        ic_search_region: Tuple[float, float],
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
            acc_sec = acc_pa[start:end]
            gyr_grad = np.gradient(gyr_ml_sec)
            ic_events.append(start + _detect_ic(gyr_ml_sec, acc_sec, gyr_grad, ic_search_region))
            fc_events.append(start + _detect_tc(gyr_ml_sec))
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
    energy = sliding_window_view(energy, window_length=min_vel_search_win_size, overlap=min_vel_search_win_size - 1)
    # find window with lowest summed energy
    min_vel_start = np.argmin(np.sum(energy, axis=1))
    # min_vel event = middle of this window
    min_vel_center = min_vel_start + min_vel_search_win_size // 2
    return min_vel_center


def _detect_ic(
    gyr_ml: np.ndarray, acc_pa: np.ndarray, gyr_ml_grad: np.ndarray, ic_search_region: Tuple[float, float],
) -> float:
    # Determine rough search region
    search_region = (np.argmax(gyr_ml), int(0.6 * len(gyr_ml)))

    if search_region[1] - search_region[0] <= 0:
        # The gyr argmax was not found in the first half of the step
        return np.nan

    # alternative:
    # refined_search_region_start, refined_search_region_end = search_region
    refined_search_region_start = search_region[0] + np.argmin(gyr_ml_grad[slice(*search_region)])
    refined_search_region_end = refined_search_region_start + np.argmax(
        gyr_ml_grad[refined_search_region_start : search_region[1]]
    )

    if refined_search_region_end - refined_search_region_start <= 0:
        return np.nan

    # Find heel strike candidate in search region based on gyr
    heel_strike_candidate = refined_search_region_start + np.argmin(
        gyr_ml[refined_search_region_start:refined_search_region_end]
    )

    # Acc search window
    acc_search_region_start = int(np.max(np.array([0, heel_strike_candidate - ic_search_region[0]])))
    acc_search_region_end = int(np.min(np.array([len(acc_pa), heel_strike_candidate + ic_search_region[1]])))

    return acc_search_region_start + np.argmin(acc_pa[acc_search_region_start:acc_search_region_end])


def _detect_tc(gyr_ml: np.ndarray) -> float:
    return np.where(np.diff(np.signbit(gyr_ml)))[0][0]

