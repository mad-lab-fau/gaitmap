"""The event detection algorithm by Rampp et al. 2014."""
from typing import Optional, Tuple, Union

import numpy as np
from numpy.linalg import norm

import pandas as pd

from gaitmap.base import BaseEventDetection, BaseType
from gaitmap.utils.consts import BF_ACC, BF_GYR
from gaitmap.utils.array_handling import sliding_window_view
from gaitmap.utils.dataset_helper import Dataset


class RamppEventDetection(BaseEventDetection):
    """Find gait events in the IMU raw signal based on signal characteristics like peaks.

    BarthDtw uses a manually created template of an IMU stride to find multiple occurrences of similar signals in a
    continuous data stream.
    The method is not limited to a single sensor-axis or sensor, but can use any number of dimensions of the provided
    input signal simultaneously.
    For more details refer to the `Notes` section.

    Attributes
    ----------
    TODO add attributes

    Parameters
    ----------
    TODO add parameters

    Other Parameters
    ----------------
    TODO add other parameters

    Notes
    -----
    TODO: Add additional details about the algorithm for event detection

    [1] Rampp, A., Barth, J., Schülein, S., Gaßmann, K. G., Klucken, J., & Eskofier, B. M. (2014). Inertial
    sensor-based stride parameter calculation from gait sequences in geriatric patients. IEEE transactions on biomedical
    engineering, 62(4), 1089-1097.. https://doi.org/10.1109/TBME.2014.2368211

    """

    ic_search_region: Tuple[float, float]
    min_vel_search_wind_size: float

    tc_: Optional[np.ndarray] = None
    min_vel_: Optional[np.ndarray] = None
    ic_: Optional[np.ndarray] = None
    pre_ic_: Optional[np.ndarray] = None
    s_id: Optional[np.ndarray] = None
    start: Optional[np.ndarray] = None
    end: Optional[np.ndarray] = None
    stride_events: pd.DataFrame = None

    data: pd.DataFrame
    sampling_rate_hz: float
    segmented_stride_list: pd.DataFrame

    def __init__(self, ic_search_region: Tuple[float, float] = (80, 50), min_vel_search_wind_size: float = 100):
        self.ic_search_region = ic_search_region
        self.min_vel_search_wind_size = min_vel_search_wind_size

    def detect(
        self: BaseType, data: Union[pd.DataFrame, Dataset], sampling_rate_hz: float, segmented_stride_list: pd.DataFrame
    ) -> BaseType:
        """Find gait events in data within strides provided by segmented_stride_list.

        Parameters
        ----------
        TODO parameters
        data
            raw data
        sampling_rate_hz
            The sampling rate of the data signal. This will be used to convert all parameters provided in seconds into
            a number of samples and it will be used to resample the template if `resample_template` is `True`.
        segmented_stride_list
            stride list from stride segmentation

        Returns
        -------
            self
                The class instance with all result attributes populated

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.segmented_stride_list = segmented_stride_list

        ic_search_region = tuple(int(v / 1000 * self.sampling_rate_hz) for v in self.ic_search_region)
        min_vel_search_wind_size = int(self.min_vel_search_wind_size / 1000 * self.sampling_rate_hz)

        acc = data[BF_ACC]
        gyr = data[BF_GYR]

        self.ic_, self.tc_, self.min_vel_ = self._find_all_events(
            gyr, acc, self.segmented_stride_list, ic_search_region, min_vel_search_wind_size
        )

        # output will have one stride less than segmented stride list
        self.s_id = np.arange(len(self.segmented_stride_list) - 1)
        self.start = self.min_vel_[:-1]
        self.end = self.min_vel_[1:]
        self.min_vel_ = self.min_vel_[:-1]
        self.pre_ic_ = self.ic_[:-1]
        self.ic_ = self.ic_[1:]
        self.tc_ = self.tc_[1:]
        stride_event_dict = {
            "s_id": self.s_id,
            "start": self.start,
            "end": self.end,
            "ic": self.ic_,
            "tc": self.tc_,
            "min_vel": self.min_vel_,
            "pre_ic": self.pre_ic_,
        }
        self.stride_events = pd.DataFrame(stride_event_dict)

        return self

    @staticmethod
    def _find_all_events(
        gyr: pd.DataFrame,
        acc: pd.DataFrame,
        stride_list: pd.DataFrame,
        ic_search_region: Tuple[float, float],
        min_vel_search_wind_size: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gyr_ml = gyr["gyr_ml"].to_numpy()
        gyr = gyr.to_numpy()
        acc_pa = acc["acc_pa"].to_numpy()
        ic_events = []
        fc_events = []
        min_vel_events = []
        for _, stride in stride_list.iterrows():
            start = stride["start"]
            end = stride["stop"]
            gyr_sec = gyr[start:end]
            gyr_ml_sec = gyr_ml[start:end]
            acc_sec = acc_pa[start:end]
            gyr_grad = np.gradient(gyr_ml_sec)
            ic_events.append(start + _detect_ic(gyr_ml_sec, acc_sec, gyr_grad, ic_search_region))
            fc_events.append(start + _detect_tc(gyr_sec))
            min_vel_events.append(start + _detect_min_vel(gyr_sec, min_vel_search_wind_size))

        return np.array(ic_events, dtype=float), np.array(fc_events, dtype=float), np.array(min_vel_events, dtype=float)


def _detect_min_vel(gyr: np.ndarray, window_size: int) -> float:
    energy = norm(gyr, axis=-1) ** 2
    energy = sliding_window_view(energy, window_length=window_size, overlap=window_size - 1)
    # find window with lowest summed energy
    min_vel_start = np.argmin(np.sum(energy, axis=1))
    # min_vel event = middle of this window
    min_vel_center = min_vel_start + window_size // 2
    return min_vel_center


def _detect_ic(
    gyr_ml: np.ndarray, acc_pa: np.ndarray, gyr_ml_grad: np.ndarray, search_window: Tuple[float, float],
) -> float:

    # Determine rough search region
    search_region = (np.argmax(gyr_ml), int(0.6 * len(gyr_ml)))

    if search_region[1] - search_region[0] <= 0:
        # The gyr argmax was not found in the first half of the step
        return np.nan

    # TODO: Redefine search region (does not work sometimes
    # alternative:
    # refined_search_region_start, refined_search_region_stop = search_region
    refined_search_region_start = search_region[0] + np.argmin(gyr_ml_grad[slice(*search_region)])
    refined_search_region_stop = refined_search_region_start + np.argmax(
        gyr_ml_grad[refined_search_region_start : search_region[1]]
    )

    if refined_search_region_stop - refined_search_region_start <= 0:
        return np.nan

    # Find heel strike candidate in search region based on gyr
    heel_strike_candidate = refined_search_region_start + np.argmin(
        gyr_ml[refined_search_region_start:refined_search_region_stop]
    )

    # Acc search window
    acc_search_region_start = int(np.max(np.array([0, heel_strike_candidate - search_window[0]])))
    acc_search_region_stop = int(np.min(np.array([len(acc_pa), heel_strike_candidate + search_window[1]])))

    return acc_search_region_start + np.argmin(acc_pa[acc_search_region_start:acc_search_region_stop])


def _detect_tc(gyr_ml: np.ndarray) -> float:
    return np.where(np.diff(np.signbit(gyr_ml)))[0][0]
