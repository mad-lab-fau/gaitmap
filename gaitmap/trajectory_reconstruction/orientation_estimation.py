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
        self.estimated_orientations_ = []
        self.estimated_orientations_.append(self.initial_orientation)
        for i_sample in range(1, len(sensor_data)):
            self.estimated_orientations_.append(
                self._next_quaternion(
                    self.estimated_orientations_[i_sample - 1, :], sensor_data[SF_GYR].iloc[i_sample, :]
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

    @staticmethod
    def _exp(q):
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
