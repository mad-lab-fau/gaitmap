"""Estimation of orientations by using inertial sensor data."""
import operator
import warnings
from itertools import accumulate
from typing import Dict

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from gaitmap.base import BaseOrientationEstimation, BaseType
from gaitmap.utils.consts import SF_GYR, SF_ACC
from gaitmap.utils.dataset_helper import (
    SingleSensorDataset,
    get_multi_sensor_dataset_names,
    Dataset,
    StrideList,
    is_single_sensor_stride_list,
    is_multi_sensor_stride_list,
)
from gaitmap.utils.dataset_helper import is_single_sensor_dataset, is_multi_sensor_dataset
from gaitmap.utils.rotations import get_gravity_rotation


class GyroIntegration(BaseOrientationEstimation):
    """Estimate orientation based on initial orientation from acc and subsequent orientation from gyr data.

    Initial orientation is calculated by aligning acceleration data of the first samples of every stride with gravity.
    Subsequent orientations are estimated by integrating gyroscope data with respect to time.

    Attributes
    ----------
    estimated_orientations_
        The first orientation is obtained by aligning the acceleration data at the start of each stride with gravity.
        Therefore, +-`half_window_width` around each start of the stride are used as we assume the start of the stride
        to have minimal velocity.
        The subsequent orientations are obtained from integrating all `len(self.data)` gyroscope samples and therefore
        this attribute contains `len(self.data) + 1` orientations
    estimated_orientations_without_initial_
        Contains `estimated_orientations_` but for each stride, the INITIAL rotation is REMOVED to make it the same
        length as `len(self.data)`.
    estimated_orientations_without_final_
        Contains `estimated_orientations_` but for each stride, the FINAL rotation is REMOVED to make it the same
        length as `len(self.data)`

    Parameters
    ----------
    align_window_width
        This is the width of the window that will be used to align the beginning of the signal of each stride with
        gravity. To do so, half the window size before and half the window size after the start of the stride will
        be used to obtain the median value of acceleration data in this phase.
        Note, that +-`np.floor(align_window_size/2)` around the start sample will be used for this. For the first
        stride, start of the stride might coincide with the start of the signal. In that case the start of the window
        would result in a negative index, thus the window to get the initial orientation will be reduced (from 0 to
        `start+np.floor(align_window_size/2)`)

    Other Parameters
    ----------------
    sensor_data
        contains gyroscope and acceleration data
    stride_event_list
        A Stride list that will be used to separate `self.data` for integration. For each stride, one sequence of
        orientations will be obtained, all of them result in a Multiindex Pandas Dataframe.
    sampling_rate_hz
        sampling rate of gyroscope data in Hz

    Examples
    --------
    >>> data = healthy_example_imu_data["left_sensor"]
    >>> stride_event_list = healthy_example_stride_event_list["left_sensor"]
    >>> gyr_int = GyroIntegration(align_window_width=8)
    >>> gyr_int.estimate(data, stride_events, 204.8)
    >>> gyr_int.estimated_orientations_without_final_["left_sensor"].iloc[0]
    qx    0.119273
    qy   -0.041121
    qz    0.000000
    qw    0.992010
    Name: (0, 0), dtype: float64
    >>> gyr_int.estimated_orientations_without_final_["left_sensor"]
                    qx        qy        qz        qw
    s_id   sample
    0      0        0.119273 -0.041121  0.000000  0.992010
           1        0.123864 -0.027830 -0.017385  0.991756
           2        0.131941 -0.013900 -0.034395  0.990563
    1      0        0.144485 -0.000534 -0.050761  0.988204
           1        0.162431  0.014014 -0.069777  0.984150
    ...               ...       ...       ...       ...

    """

    align_window_width: int

    data: Dataset
    event_list: StrideList
    sampling_rate_hz: float

    def __init__(self, align_window_width: int = 8):
        self.align_window_width = align_window_width

    def estimate(self: BaseType, data: Dataset, stride_event_list: StrideList, sampling_rate_hz: float) -> BaseType:
        """Use the initial rotation and the gyroscope signal to estimate the orientation to every time point .

        Parameters
        ----------
        data
            At least must contain 3D-gyroscope data.
        stride_event_list
            List of events for one or multiple sensors.
        sampling_rate_hz
            Sampling rate with which gyroscopic data was recorded.

        Notes
        -----
        This function makes use of :py:meth:`~scipy.spatial.transform.Rotation.from_rotvec` to turn the gyro signal of
        each sample into a differential quaternion.
        This means that the rotation between two samples is assumed to be constant around one axis.
        The initial orientation is obtained by aligning acceleration data in the beginning of the signal with gravity.

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.event_list = stride_event_list

        if not is_single_sensor_stride_list(
            stride_event_list, stride_type="min_vel"
        ) and not is_multi_sensor_stride_list(stride_event_list, stride_type="min_vel"):
            raise ValueError("Provided stride event list is not supported by gaitmap")

        if is_single_sensor_dataset(self.data):
            self.estimated_orientations_ = self._estimate_single_sensor(self.data, self.event_list)
        elif is_multi_sensor_dataset(self.data):
            self.estimated_orientations_ = self._estimate_multi_sensor()
        else:
            raise ValueError("Provided data set is not supported by gaitmap")
        return self

    def _estimate_single_sensor(self, data: SingleSensorDataset, event_list: StrideList) -> pd.DataFrame:
        cols = ["qx", "qy", "qz", "qw"]
        rotations = dict()
        for i_s_id, i_stride in event_list.iterrows():
            i_start, i_end = (int(i_stride["start"]), int(i_stride["end"]))
            i_rotations = self._estimate_stride(data, i_start, i_end)
            rotations[i_s_id] = pd.DataFrame(i_rotations.as_quat(), columns=cols)
        rotations = pd.concat(rotations)
        rotations.index = rotations.index.rename(("s_id", "sample"))
        return rotations

    def _estimate_stride(self, data: SingleSensorDataset, start: int, end: int) -> Rotation:
        initial_orientation = self._calculate_initial_orientation(data, start)
        gyro_data = data[SF_GYR].iloc[start:end].to_numpy()
        single_step_rotations = Rotation.from_rotvec(gyro_data / self.sampling_rate_hz)
        # This is faster than np.cumprod. Custom quat rotation would be even faster, as we could skip the second loop
        out = accumulate([initial_orientation, *single_step_rotations], operator.mul)
        return Rotation([o.as_quat() for o in out])

    def _estimate_multi_sensor(self) -> Dict[str, pd.DataFrame]:
        orientations = dict()
        for i_sensor in get_multi_sensor_dataset_names(self.data):
            orientations[i_sensor] = self._estimate_single_sensor(self.data[i_sensor], self.event_list[i_sensor])
        return orientations

    def _calculate_initial_orientation(self, data: SingleSensorDataset, start) -> Rotation:
        half_window = int(np.floor(self.align_window_width / 2))
        if start - half_window >= 0:
            start_sample = start - half_window
        else:
            start_sample = 0
            warnings.warn("Could not use complete window length for initializing orientation.")
        if start + half_window < len(data):
            end_sample = start + half_window
        else:
            end_sample = len(data) - 1
            warnings.warn("Could not use complete window length for initializing orientation.")
        acc = (data[SF_ACC].iloc[start_sample:end_sample]).median()
        acc_normalized = acc / np.linalg.norm(acc.values, 2)
        # get_gravity_rotation assumes [0, 0, 1] as gravity
        return get_gravity_rotation(acc_normalized)
