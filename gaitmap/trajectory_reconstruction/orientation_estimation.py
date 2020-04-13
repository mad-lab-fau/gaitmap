"""Estimation of orientations by gyroscope integration."""
import operator
from itertools import accumulate
from typing import Union, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from gaitmap.base import BaseOrientationEstimation
from gaitmap.utils.consts import SF_GYR, SF_ACC
from gaitmap.utils.dataset_helper import (
    SingleSensorDataset,
    get_multi_sensor_dataset_names,
    Dataset,
    StrideList,
)
from gaitmap.utils.dataset_helper import is_single_sensor_dataset, is_multi_sensor_dataset
from gaitmap.utils.rotations import get_gravity_rotation


class GyroIntegration(BaseOrientationEstimation):
    """Estimate orientation based on a given initial orientation.

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

    Parameters
    ----------
    align_window_width
        This is the width of the window that will be used to align the beginning of the signal of each stride with
        gravity. To do so, half the window size before and half the window size after the start of the stride will
        be used to obtain the median value of acceleration data in this phase.
        Note, that +-`np.floor(align_window_size/2)` around the start sample will be used for this.

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
    # single sensor
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
    """

    align_window_width: int
    data: Dataset
    event_list = StrideList
    sampling_rate_hz: float

    def __init__(self, align_window_width):
        self.align_window_width = align_window_width
        pass

    def estimate(self, data: Dataset, stride_event_list: StrideList, sampling_rate_hz: float):
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
        This function makes use of `from_rotvec` of :func:`~scipy.spatial.transform.Rotation`, to turn the gyro signal
        of each sample into a differential quaternion.
        This means that the rotation between two samples is assumed to be constant around one axis.
        The initial orientation is obtained by aligning acceleration data in the beginning of the signal with gravity.

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.event_list = stride_event_list

        if is_single_sensor_dataset(self.data):
            self.estimated_orientations_ = self._estimate_single_sensor(self.data, self.event_list)
        elif is_multi_sensor_dataset(self.data):
            self.estimated_orientations_ = self._estimate_multi_sensor()
        else:
            raise ValueError("Provided data set is not supported by gaitmap")
        return self

    def _estimate_single_sensor(self, data: SingleSensorDataset, event_list: StrideList) -> Tuple[Rotation, Rotation]:
        # TODO: put cols into consts?
        cols = ["s_id", "qx", "qy", "qz", "qw"]
        rotations = pd.DataFrame(columns=cols)
        for i_s_id, i_stride in event_list.iterrows():
            # TODO: rework start and end to min_vel?
            i_start, i_end = (int(i_stride["start"]), int(i_stride["end"]))
            i_rotations = self._estimate_stride(data, i_start, i_end)
            i_rotations_pd = pd.DataFrame(i_rotations.as_quat(), columns=cols[1:])
            i_rotations_pd["s_id"] = i_s_id
            rotations = rotations.append(i_rotations_pd)
        # TODO: Compare index names with position and dataset_helper
        rotations.index.rename("sample", inplace=True)
        return rotations.set_index("s_id", append=True)

    def _estimate_stride(self, data: SingleSensorDataset, start: int, end: int) -> Rotation:
        initial_orientation = self._calculate_initial_orientation(data, start)
        gyro_data = data[SF_GYR].iloc[start:end].to_numpy()
        single_step_rotations = Rotation.from_rotvec(gyro_data / self.sampling_rate_hz)
        # This is faster than np.cumprod. Custom quat rotation would be even faster, as we could skip the second loop
        out = accumulate([initial_orientation, *single_step_rotations], operator.mul)
        out_as_rot = Rotation([o.as_quat() for o in out])
        return out_as_rot

    def _estimate_multi_sensor(self) -> Tuple[Dict[str, Rotation], Dict[str, Rotation]]:
        orientations = dict()
        for i_sensor in get_multi_sensor_dataset_names(self.data):
            ori = self._estimate_single_sensor(self.data[i_sensor], self.event_list[i_sensor])
            orientations[i_sensor] = ori
        return orientations

    def _calculate_initial_orientation(self, data: SingleSensorDataset, start):
        half_window = int(np.floor(self.align_window_width / 2))
        acc = (data[SF_ACC].iloc[start - half_window : start + half_window]).median()
        acc_normalized = acc / np.linalg.norm(acc.values, 2)
        # get_gravity_rotation assumes [0, 0, 1] as gravity
        return get_gravity_rotation(acc_normalized)
