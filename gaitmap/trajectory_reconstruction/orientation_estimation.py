"""Estimation of orientations by trapezoidal gyroscope integration."""
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from gaitmap.base import BaseOrientationEstimation
from gaitmap.utils.consts import SF_GYR


class GyroIntegration(BaseOrientationEstimation):
    """Estimate orientation based on a given initial orientation.

    Subsequent orientations are estimated by integrating gyroscope data with respect to time. Calculations are performed
    by Sabatini et al. [1].

    Parameters
    ----------
    initial_orientation
        Specifies what the initial orientation of the sensor is. Subsequent rotations will be estimated using this
        rotation is initial state.

    Attributes
    ----------
    estimated_orientations_
        Contains rotations based on initial rotation and trapezoidal gyroscope integration. The first value of this
        array is the initial rotation, passed to the __init__.

    Other Parameters
    ----------------
    sensor_data
        contains gyroscope and acceleration data
    sampling_rate_hz : float
        sampling rate of gyroscope data in Hz

    Notes
    -----
    [1] Sabatini, A. M. (2005). Quaternion-based strap-down integration method for applications of inertial sensing to
    gait analysis. Medical & biological engineering & computing 43 (1), 94–101. https://doi.org/10.1007/BF02345128
    We perform the calulation a bit differently s.t. we don't need to caluclate e^(OMEGA) (eq. 3 by Sabatini)

    """

    sensor_data: pd.DataFrame
    estimated_orientations_: list
    sampling_rate_hz: float
    initial_orientation: Rotation

    def __init__(self, initial_orientation: Rotation):
        # TODO: check: relevant if rotation is world->sensor or sensor->world? I don't think so
        self.initial_orientation = initial_orientation

    def estimate_orientation_sequence(self, sensor_data: pd.DataFrame, sampling_rate_hz: float):
        """Use the initial rotation and the gyroscope signal to estimate rotations of second until last sample.

        Parameters
        ----------
        sensor_data
            At least must contain 3D-gyroscope data for one sensor (multiple sensors not yet implemented).
        sampling_rate_hz
            Sampling rate with which gyroscopic data was recorded.

        """
        # TODO: so far it is only possible to pass one sensor with columns being gyr_x...acc_z
        #  ==> adapt to multiple sensors
        self.sampling_rate_hz = sampling_rate_hz
        self.estimated_orientations_ = []
        self.estimated_orientations_.append(self.initial_orientation)
        print('TEST: ', len(sensor_data))
        for i_sample in range(1, len(sensor_data)+1):
            self.estimated_orientations_.append(
                self._next_quaternion(
                    self.estimated_orientations_[i_sample - 1], sensor_data[SF_GYR].iloc[i_sample-1]
                )
            )

    def _next_quaternion(self, previous_quaternion, gyr: pd.Series) -> Rotation:
        # TODO: Check if gyroscope data is in degeree or rad
        x = 0
        y = 1
        z = 2

        diff_quaternion_gyro = np.multiply(1 / (2 * self.sampling_rate_hz), [0, gyr[x], gyr[y], gyr[z]])
        diff_quaternion_gyro = self._exp(diff_quaternion_gyro)
        return Rotation(previous_quaternion * diff_quaternion_gyro)

    #@staticmethod
    def _exp(self, q: np.ndarray) -> np.ndarray:
        # TODO: move to utils

        quaternion = np.zeros(4)
        multiplication_factor = np.exp(q[0])
        squared_helper = np.sqrt(q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
        quaternion[0] = multiplication_factor * np.cos(squared_helper)
        quaternion[1] = q[1] * multiplication_factor * np.sin(squared_helper) / squared_helper
        quaternion[2] = q[2] * multiplication_factor * np.sin(squared_helper) / squared_helper
        quaternion[3] = q[3] * multiplication_factor * np.sin(squared_helper) / squared_helper
        quaternion = quaternion / np.linalg.norm(quaternion, 2)
        return quaternion

    @staticmethod
    def quaternion_multiply(quat0: list, quat1: list):
        p0, p1, p2, p3 = quat0
        q0, q1, q2, q3 = quat1
        w0, x0, y0, z0 = quat0
        w1, x1, y1, z1 = quat1
        result = np.ndarray([p0*q0 - p1*q1 - p2*q2 - p3*q3,
                             p1*q0 + p0*q1 + p2*q3 - p3*q2,
                             p2*q0 + p0*q2 + p3*q1 - p1*q3,
                             p3*q0 + p0*q3 + p1*q2 - p2*q1])

       # result = np.array([-x0 * x1 - y1 * y0 - z1 * z0 + w1 * w0,
       #                  x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
       #                  -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
       #                  x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)
        return np.divide(result, np.linalg.norm(result, 2))
