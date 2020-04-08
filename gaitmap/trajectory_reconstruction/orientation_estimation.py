"""Estimation of orientations by gyroscope integration."""
import operator
from itertools import accumulate
from typing import Union, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from gaitmap.base import BaseOrientationEstimation
from gaitmap.utils.consts import SF_GYR, SF_POS, SF_VEL, SF_ACC
from gaitmap.utils.dataset_helper import (
    SingleSensorDataset,
    MultiSensorDataset,
    get_multi_sensor_dataset_names,
    SingleSensorStrideList,
    MultiSensorStrideList,
    Dataset,
    StrideList,
)
from gaitmap.utils.dataset_helper import is_single_sensor_dataset, is_multi_sensor_dataset
from gaitmap.utils.rotations import find_shortest_rotation


class GyroIntegration(BaseOrientationEstimation):
    """Estimate orientation based on a given initial orientation.

    Subsequent orientations are estimated by integrating gyroscope data with respect to time.

    Parameters
    ----------
    initial_orientation
        Specifies what the initial orientation of the sensor is. Subsequent rotations will be estimated using this
        rotation is initial state.

    Attributes
    ----------
    estimated_orientations_
        Contains the resulting orientations (represented as a :class:`~scipy.spatial.transform.Rotation` object).
        This has the same length than the passed input data (i.e. the initial orientation is not included)
    estimated_orientations_with_initial_
        Same as `estimated_orientations_` but contains the initial orientation as the first index.
        Therefore, it contains one more sample than the input data.

    Other Parameters
    ----------------
    sensor_data
        contains gyroscope and acceleration data
    sampling_rate_hz : float
        sampling rate of gyroscope data in Hz

    Examples
    --------
    # single sensor
    >>> gyr_integrator = GyroIntegration()
    >>> gyr_integrator.estimate(data, 204.8)
    >>> orientations = gyr_integrator.estimated_orientations_
    >>> orientations[-1].as_quat()
    array([0., 1, 0., 0.])

    """

    initial_orientation: Union[Rotation, Dict[str, Rotation]]

    estimated_orientations_: Union[Rotation, Dict[str, Rotation]]
    estimated_orientations_with_initial_: Union[Rotation, Dict[str, Rotation]]

    data: Dataset
    sampling_rate_hz: float
    stride_list = StrideList

    def __init__(self):
        # Originally we planned to write a wrapper, which calls this class for every stride, and thus we would have
        # needed the initial orientation here. For now, we calculate the initial orientation in this class here.
        # self.initial_orientation = initial_orientation
        pass

    def estimate(self, data: Dataset, stride_list: StrideList, sampling_rate_hz: float):
        """Use the initial rotation and the gyroscope signal to estimate the orientation to every time point .

        Parameters
        ----------
        data
            At least must contain 3D-gyroscope data.
        stride_list
            List of strides for one or multiple sensors.
        sampling_rate_hz
            Sampling rate with which gyroscopic data was recorded.

        Notes
        -----
        This function makes use of `from_rotvec` of :func:`~scipy.spatial.transform.Rotation`, to turn the gyro signal
        of each sample into a differential quaternion.
        This means that the rotation between two samples is assumed to be constant around one axis.

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.stride_list = stride_list

        if is_single_sensor_dataset(self.data):
            self.estimated_orientations_, self.estimated_orientations_with_initial_ = self._estimate_single_sensor(
                self.data, self.stride_list
            )
        elif is_multi_sensor_dataset(self.data):
            self.estimated_orientations_, self.estimated_orientations_with_initial_ = self._estimate_multi_sensor(data)
        else:
            raise ValueError("Provided data set is not supported by gaitmap")
        return self

    def _estimate_single_sensor(self, data: SingleSensorDataset, event_list: StrideList) -> Tuple[Rotation, Rotation]:
        # TODO: put cols into consts?
        cols = ["s_id", "qx", "qy", "qz", "qw"]
        rotations_without_initial = pd.DataFrame(columns=cols)
        rotations = pd.DataFrame(columns=cols)
        for i_s_id, i_stride in event_list.iterrows():
            # TODO: check it this loop can be simplified
            # TODO: rework start and end to min_vel?
            i_start, i_end = (int(i_stride["start"]), int(i_stride["end"]))
            i_rotations = self._estimate_stride(data.iloc[i_start:i_end])
            i_rotations_pd = pd.DataFrame(i_rotations.as_quat(), columns=cols[1:])
            i_rotations_pd["s_id"] = i_s_id
            rotations_without_initial = rotations_without_initial.append(i_rotations_pd.iloc[1:])
            rotations = rotations.append(i_rotations_pd)
        return rotations_without_initial.set_index("s_id", append=True), rotations.set_index("s_id", append=True)

    def _estimate_stride(self, data: SingleSensorDataset) -> Rotation:
        gyro_data = data[SF_GYR].to_numpy()
        initial_orientation = self._calculate_initial_orientation(data)
        single_step_rotations = Rotation.from_rotvec(gyro_data / self.sampling_rate_hz)
        # This is faster than np.cumprod. Custom quat rotation would be even faster, as we could skip the second loop
        out = accumulate([initial_orientation, *single_step_rotations], operator.mul)
        out_as_rot = Rotation([o.as_quat() for o in out])
        return out_as_rot

    def _estimate_multi_sensor(self, data: MultiSensorDataset) -> Tuple[Dict[str, Rotation], Dict[str, Rotation]]:
        estimated_orientations_ = dict()
        estimated_ori_with_initial_ = dict()
        for sensor in get_multi_sensor_dataset_names(data):
            ori, with_initial = self._estimate_single_sensor(data[sensor])
            estimated_orientations_[sensor] = ori
            estimated_ori_with_initial_[sensor] = with_initial
        return estimated_orientations_, estimated_ori_with_initial_

    def _calculate_initial_orientation(self, data: SingleSensorDataset):
        # TODO: change this, s.t. we use a window of the acc signal instead of relying on only one, possibly noisy,
        #  sample
        acc = data[SF_ACC].iloc[0]
        # TODO: put to consts
        gravity = np.array([0, 0, 1])
        acc_normalized = acc/np.linalg.norm(acc.values, 2)
        return find_shortest_rotation(acc_normalized, gravity)
