"""Madgwick Cython Implimentation

Direct adoption of the original Madgwick code into Cython.
Original code can be found here: http://x-io.co.uk/open-source-imu-and-ahrs-algorithms/
The original code and all its direct modifications published under GNU-GPL.
"""
from typing import Union

import numpy as np
from numba import njit
from scipy.spatial.transform import Rotation

from gaitmap.base import BaseOrientationMethods, BaseType
from gaitmap.utils.consts import SF_GYR, SF_ACC
from gaitmap.utils.dataset_helper import SingleSensorDataset, is_single_sensor_dataset
from gaitmap.utils.fast_quaternion_math import rate_of_change_from_gyro


class Madgwick(BaseOrientationMethods):
    initial_orientation: Union[np.ndarray, Rotation]
    beta: float

    orientations_: Rotation

    data: SingleSensorDataset
    sampling_rate_hz: float

    def __init__(self, beta: float = 0.1, initial_orientation: Union[np.ndarray, Rotation] = Rotation.identity()):
        self.initial_orientation = initial_orientation
        self.beta = beta

    # TODO: Allow to continue the integration
    # TODO: Clarify the expected input unit!
    def estimate(self: BaseType, data: SingleSensorDataset, sampling_rate_hz: float) -> BaseType:
        """Estimate the orientation of the sensor by simple integration of the Gyro data.

        Parameters
        ----------
        data
            Continous sensor data that includes at least a Gyro
        sampling_rate_hz
            The sampling rate of the data in Hz

        Returns
        -------
        self
            The class instance with all result attributes populated

        """
        if not is_single_sensor_dataset(data, frame="sensor"):
            raise ValueError("Data is not a single sensor dataset.")
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        initial_orientation = self.initial_orientation
        if isinstance(initial_orientation, Rotation):
            initial_orientation = Rotation.as_quat(initial_orientation)
        gyro_data = np.deg2rad(data[SF_GYR].to_numpy())
        acc_data = data[SF_ACC].to_numpy()

        rots = _madgwick_update_series(
            gyro=gyro_data,
            acc=acc_data,
            initial_orientation=initial_orientation,
            sampling_rate_hz=sampling_rate_hz,
            beta=self.beta,
        )
        self.orientations_ = Rotation.from_quat(rots)
        return self


@njit()
def _madgwick_update(gyro, acc, initial_orientation, sampling_rate_hz, beta):
    q = initial_orientation

    qdot = rate_of_change_from_gyro(gyro, q)

    if np.any(acc != 0):
        acc /= np.sqrt(np.sum(acc ** 2))
        ax, ay, az = acc

        # Note that we change the order of q components here as we use a different quaternion definition.
        q0 = q[3]
        q1, q2, q3 = q[:3]

        # Auxiliary variables to avoid repeated arithmetic
        _2q0 = 2.0 * q0
        _2q1 = 2.0 * q1
        _2q2 = 2.0 * q2
        _2q3 = 2.0 * q3
        _4q0 = 4.0 * q0
        _4q1 = 4.0 * q1
        _4q2 = 4.0 * q2
        _8q1 = 8.0 * q1
        _8q2 = 8.0 * q2
        q0q0, q1q1, q2q2, q3q3 = q ** 2

        # Gradient decent algorithm corrective step
        s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay
        s1 = _4q1 * q3q3 - _2q3 * ax + 4.0 * q0q0 * q1 - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az
        s2 = 4.0 * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az
        s3 = 4.0 * q1q1 * q3 - _2q1 * ax + 4.0 * q2q2 * q3 - _2q2 * ay

        # Switch the component order back
        s = np.array([s1, s2, s3, s0])
        s /= np.sqrt(np.sum(s ** 2))

        # Apply feedback step
        qdot -= beta * s

    # Integrate rate of change of quaternion to yield quaternion
    q += qdot / sampling_rate_hz
    q /= np.sqrt(np.sum(q ** 2))

    return q


@njit()
def _madgwick_update_series(gyro, acc, initial_orientation, sampling_rate_hz, beta):
    out = np.empty((len(gyro) + 1, 4))
    out[0] = initial_orientation
    for i in range(len(gyro)):
        initial_orientation = _madgwick_update(
            gyro=gyro[i],
            acc=acc[i],
            initial_orientation=initial_orientation,
            sampling_rate_hz=sampling_rate_hz,
            beta=beta,
        )
        out[i + 1] = initial_orientation

    return out
