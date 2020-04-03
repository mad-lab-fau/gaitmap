"""The event detection algorithm by Rampp et al. 2014."""
from typing import Optional, Sequence, List

from typing import Optional, Tuple, List

import numpy as np
from numpy.linalg import norm

import pandas as pd

from slither.imu.utils.static_moment_detection import sliding_window_view


from gaitmap.base import BaseEventDetection, BaseType


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
    TODO add other paramters

    Notes
    -----
    TODO: Add additional details about the use of DTW for stride segmentation

    [1] Rampp, A., Barth, J., Schülein, S., Gaßmann, K. G., Klucken, J., & Eskofier, B. M. (2014). Inertial
    sensor-based stride parameter calculation from gait sequences in geriatric patients. IEEE transactions on biomedical
    engineering, 62(4), 1089-1097.. https://doi.org/10.1109/TBME.2014.2368211

    """

    ic_search_region: Tuple[float, float]
    min_vel_search_wind_size: float

    tc_: Optional[np.ndarray] = None
    min_vel_: Optional[np.ndarray] = None
    ic_: Optional[np.ndarray] = None

    data: pd.DataFrame
    sampling_rate_hz: float
    segmented_stride_list: pd.DataFrame

    def __init__(
        self,
        ic_search_region: Tuple[float, float] = None,
        min_vel_search_wind_size: float = None
    ):
        self.ic_search_region = ic_search_region
        self.min_vel_search_wind_size = min_vel_search_wind_size

    def detect(
        self: BaseType, data: pd.DataFrame, sampling_rate_hz: float, segmented_stride_list: pd.DataFrame
    ) -> BaseType:
        """Find gait events in data within strides provided by segmented_stride_list.

    Parameters
    ----------
    TODO parameters
    data
    sampling_rate_hz
        The sampling rate of the data signal. This will be used to convert all parameters provided in seconds into
        a number of samples and it will be used to resample the template if `resample_template` is `True`.
    segmented_stride_list

    Returns
    -------
        self
            The class instance with all result attributes populated

    """

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.segmented_stride_list = segmented_stride_list

        ic_search_region = tuple(int(v / 1000 * self.sampling_rate_hz) for v in self.ic_search_region)
        min_vel_search_wind_size = int(self.min_vel_search_wind_size / 1000 * self.sampling_rate)
        self.ic_, self.tc_, self.min_vel_ = self._find_all_events(
            gyro, acc, self.segmented_stride_list, ic_search_region, min_vel_search_wind_size
        )

        return self

    @staticmethod
    def _find_all_events(gyro, acc, stride_list, hs_search_region, ms_search_wind_size):
        gyr_saggital = gyro[:, 1]
        acc_ant_post = acc[:, 0]
        hs_events = []
        to_events = []
        ms_events = []
        for i in range(len(stride_list)):
            start = stride_list[i][0]
            end = stride_list[i][1]
            gyr_sec = gyro[start:end]
            gyr_saggital_sec = gyr_saggital[start:end]
            acc_sec = acc_ant_post[start:end]
            gyr_grad = np.gradient(gyr_sec)
            hs_events.append(start + _find_heel_strike(gyr_saggital_sec, acc_sec, gyr_grad, hs_search_region))
            to_events.append(start + _find_toe_off(gyr_sec))
            ms_events.append(start + _find_mid_stance(gyr_sec, ms_search_wind_size))

        return np.array(hs_events, dtype=float), np.array(to_events, dtype=float), np.array(ms_events, dtype=float)

    def _detect_ic(
        gyr_sagittal: np.ndarray,
        acc_ant_post: np.ndarray,
        gyr_sagittal_grad: np.ndarray,
        search_window: Tuple[float, float],
    ) -> int:

        # Determine rough search region
        search_region = (np.argmax(gyr_sagittal), int(0.6 * len(gyr_sagittal)))

        if search_region[1] - search_region[0] <= 0:
            # The gyro argmax was not found in the first half of the step
            return np.nan

        # TODO: Redefine search region (does not work sometimes
        # alternative:
        # refined_search_region_start, refined_search_region_stop = search_region
        refined_search_region_start = search_region[0] + np.argmin(gyr_sagittal_grad[slice(*search_region)])
        refined_search_region_stop = refined_search_region_start + np.argmax(
            gyr_sagittal_grad[refined_search_region_start : search_region[1]]
        )

        if refined_search_region_stop - refined_search_region_start <= 0:
            return np.nan

        # Find heel strike candidate in search region based on gyro
        heel_strike_candidate = refined_search_region_start + np.argmin(
            gyr_sagittal[refined_search_region_start:refined_search_region_stop]
        )

        # Acc search window
        acc_search_region_start = int(np.max(np.array([0, heel_strike_candidate - search_window[0]])))
        acc_search_region_stop = int(np.min(np.array([len(acc_ant_post), heel_strike_candidate + search_window[1]])))

        return acc_search_region_start + np.argmin(acc_ant_post[acc_search_region_start:acc_search_region_stop])

    def _detect_tc(gyr_sagittal: np.ndarray) -> int:
        return np.where(np.diff(np.signbit(gyr_sagittal)))[0][0]

    def _detect_min_vel(gyro: np.ndarray, window_size: int) -> int:
        energy = norm(gyro, axis=-1) ** 2
        energy = sliding_window_view(energy, shape=(window_size,))
        min_vel_start = np.nanmin(energy)
        min_vel_center = min_vel_start + window_size // 2
        return min_vel_center
