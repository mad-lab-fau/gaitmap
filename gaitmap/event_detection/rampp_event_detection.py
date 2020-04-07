"""The event detection algorithm by Rampp et al. 2014."""
from typing import Optional, Tuple, Union, Dict

import numpy as np
from numpy.linalg import norm

import pandas as pd

from gaitmap.base import BaseEventDetection, BaseType
from gaitmap.utils.consts import BF_ACC, BF_GYR
from gaitmap.utils.array_handling import sliding_window_view
from gaitmap.utils import dataset_helper
from gaitmap.utils.dataset_helper import Dataset


class RamppEventDetection(BaseEventDetection):
    """Find gait events in the IMU raw signal based on signal characteristics.

    RamppEventDetection uses signal processing approaches to find temporal gait events by searching for characteristic
    features in the signals as described in Rampp et al. (2014) [1]_.
    For more details refer to the `Notes` section.

    Parameters
    ----------
    ic_search_region_ms
        The region to look for the initial contact in the acc_pa signal in ms given an ic candidate. According to [1]_,
        for the ic the algorithm first looks for a local minimum in the gyr_ml signal after the swing phase. The actual
        ic is then determined in the acc_pa signal in the ic_search_region_ms around that gyr_ml minimum.
        ic_search_region_ms[0] describes the start and ic_search_region_ms[1] the end of the region to check around the
        gyr_ml minimum.
    min_vel_search_win_size_ms
        The size of the sliding window for finding the minimum gyroscope energy in ms

    Attributes
    ----------
    stride_events_: A stride list or dictionary with such values
        The result of the `detect` method holding all temporal gait events and start / end of all strides. Formatted
        as pandas DataFrame
    start_: 1D array or dictionary with such values
        The array of start samples of all strides
    end_: 1D array or dictionary with such values
        The array of end samples of all strides
    tc_: 1D array or dictionary with such values
        The array of terminal contact samples of all strides
    min_vel_: 1D array or dictionary with such values
        The array of min_vel samples of all strides
    ic_: 1D array or dictionary with such values
        The array of initial contact samples of all strides
    pre_ic_: 1D array or dictionary with such values
        The array of pre-initial contact samples of all strides

    Other Parameters
    ----------------
    data
        The data passed to the `detect` method.
    sampling_rate_hz
        The sampling rate of the data
    segmented_stride_list
        A list of strides provided by a stride segmentation method

    Notes
    -----
    TODO: Add additional details about the algorithm for event detection

    .. [1] Rampp, A., Barth, J., Schülein, S., Gaßmann, K. G., Klucken, J., & Eskofier, B. M. (2014). Inertial
       sensor-based stride parameter calculation from gait sequences in geriatric patients. IEEE transactions on
       biomedical engineering, 62(4), 1089-1097.. https://doi.org/10.1109/TBME.2014.2368211

    """

    ic_search_region_ms: Tuple[float, float]
    min_vel_search_win_size_ms: float

    start_: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
    end_: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
    tc_: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
    min_vel_: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
    ic_: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
    pre_ic_: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
    stride_events_: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]

    data: Dataset
    sampling_rate_hz: float
    segmented_stride_list: pd.DataFrame

    def __init__(self, ic_search_region_ms: Tuple[float, float] = (80, 50), min_vel_search_win_size_ms: float = 100):
        self.ic_search_region_ms = ic_search_region_ms
        self.min_vel_search_win_size_ms = min_vel_search_win_size_ms

    def detect(self: BaseType, data: Dataset, sampling_rate_hz: float, segmented_stride_list: pd.DataFrame) -> BaseType:
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

        Examples
        --------
        Get gait events from single sensor signal

        >>> event_detection = RamppEventDetection()
        >>> event_detection.detect(data=data_left, sampling_rate_hz=204.8, segmented_stride_list=stride_list_left)
        >>> event_detection.stride_events_
            s_id   start     end      ic      tc  min_vel  pre_ic
        0      0   519.0   710.0   651.0   584.0    519.0   498.0
        1      1   710.0   935.0   839.0   802.0    710.0   651.0
        2      2   935.0  1183.0  1089.0  1023.0    935.0   839.0
        ...

        """
        if dataset_helper.is_multi_sensor_dataset(data):
            raise NotImplementedError("Multisensor input is not supported yet")

        if not dataset_helper.is_single_sensor_dataset(data):
            raise ValueError("Provided data set is not supported by gaitmap")

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.segmented_stride_list = segmented_stride_list

        ic_search_region = tuple(int(v / 1000 * self.sampling_rate_hz) for v in self.ic_search_region_ms)
        if all(v == 0 for v in ic_search_region):
            raise ValueError("The values chosen for ic_search_region_ms are too close to zero.")
        min_vel_search_win_size_ms = int(self.min_vel_search_win_size_ms / 1000 * self.sampling_rate_hz)

        acc = data[BF_ACC]
        gyr = data[BF_GYR]

        s_id, ic, tc, min_vel = self._find_all_events(
            gyr, acc, self.segmented_stride_list, ic_search_region, min_vel_search_win_size_ms
        )

        s_id = s_id[:-1]  # ignore the last s_id as the last segmented stride is not preserved by rampp
        start = min_vel[:-1]
        end = min_vel[1:]
        min_vel = min_vel[:-1]
        pre_ic = ic[:-1]
        ic = ic[1:]
        tc = tc[1:]
        stride_event_dict = {
            "s_id": s_id,
            "seg_start": self.segmented_stride_list["start"].iloc[:-1],
            "seg_end": self.segmented_stride_list["end"].iloc[:-1],
            "start": start,
            "end": end,
            "ic": ic,
            "tc": tc,
            "min_vel": min_vel,
            "pre_ic": pre_ic,
        }

        # find breaks in segmented strides and drop first strides of new sequences
        # TODO shift this to cleanup function in the future
        stride_list_breaks = _find_breaks_in_stride_list(self.segmented_stride_list)
        stride_event_df = pd.DataFrame(stride_event_dict)
        stride_event_df = stride_event_df.drop(stride_event_df.index[stride_list_breaks])

        self.stride_events_ = stride_event_df.drop(["seg_start", "seg_end"], axis=1)
        self.start_ = self.stride_events_["start"]
        self.end_ = self.stride_events_["end"]
        self.min_vel_ = self.stride_events_["min_vel"]
        self.pre_ic_ = self.stride_events_["pre_ic"]
        self.ic_ = self.stride_events_["ic"]
        self.tc_ = self.stride_events_["tc"]

        return self

    @staticmethod
    def _find_all_events(
        gyr: pd.DataFrame,
        acc: pd.DataFrame,
        stride_list: pd.DataFrame,
        ic_search_region: Tuple[float, float],
        min_vel_search_win_size_ms: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        gyr_ml = gyr["gyr_ml"].to_numpy()
        gyr = gyr.to_numpy()
        acc_pa = -acc["acc_pa"].to_numpy()  # have to invert acc data to work on rampp paper
        s_id = []
        ic_events = []
        fc_events = []
        min_vel_events = []
        for _, stride in stride_list.iterrows():
            s_id.append(stride["s_id"])
            start = stride["start"]
            end = stride["end"]
            gyr_sec = gyr[start:end]
            gyr_ml_sec = gyr_ml[start:end]
            acc_sec = acc_pa[start:end]
            gyr_grad = np.gradient(gyr_ml_sec)
            ic_events.append(start + _detect_ic(gyr_ml_sec, acc_sec, gyr_grad, ic_search_region))
            fc_events.append(start + _detect_tc(gyr_ml_sec))
            min_vel_events.append(start + _detect_min_vel(gyr_sec, min_vel_search_win_size_ms))

        # TODO add validation / consistency check of the gait events
        return (
            np.array(s_id, dtype=int),
            np.array(ic_events, dtype=float),
            np.array(fc_events, dtype=float),
            np.array(min_vel_events, dtype=float),
        )


def _detect_min_vel(gyr: np.ndarray, min_vel_search_win_size_ms: int) -> float:
    energy = norm(gyr, axis=-1) ** 2
    if min_vel_search_win_size_ms >= len(energy):
        raise ValueError("The value chosen for min_vel_search_win_size_ms is too large. Should be 100 ms.")
    energy = sliding_window_view(
        energy, window_length=min_vel_search_win_size_ms, overlap=min_vel_search_win_size_ms - 1
    )
    # find window with lowest summed energy
    min_vel_start = np.argmin(np.sum(energy, axis=1))
    # min_vel event = middle of this window
    min_vel_center = min_vel_start + min_vel_search_win_size_ms // 2
    return min_vel_center


def _detect_ic(
    gyr_ml: np.ndarray, acc_pa: np.ndarray, gyr_ml_grad: np.ndarray, ic_search_region: Tuple[float, float],
) -> float:
    # Determine rough search region
    search_region = (np.argmax(gyr_ml), int(0.6 * len(gyr_ml)))

    if search_region[1] - search_region[0] <= 0:
        # The gyr argmax was not found in the first half of the step
        return np.nan

    # TODO: Redefine search region (does not work sometimes
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


def _find_breaks_in_stride_list(stride_list: pd.DataFrame) -> list:
    tmp = stride_list["start"].iloc[1:].to_numpy() - stride_list["end"].iloc[:-1].to_numpy()
    breaks = np.where(tmp != 0)[0] + 1
    return list(breaks)
