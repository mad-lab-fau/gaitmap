"""Estimation of orientations by gyroscope integration."""
import pandas as pd
from scipy.spatial.transform import Rotation

from gaitmap.base import BaseOrientationEstimation
from gaitmap.utils.consts import SF_GYR


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
        Contains rotations based on initial rotation and gyroscope integration. The first value of this array is the
        initial rotation, passed to the __init__. All rotations are :class:`scipy.spatial.transform.Rotation` objects.

    Other Parameters
    ----------------
    sensor_data
        contains gyroscope and acceleration data
    sampling_rate_hz : float
        sampling rate of gyroscope data in Hz

    Examples
    --------
    >>> gyr_integrator = GyroIntegration(Rotation([0, 0, 1, 0]))

    """

    sensor_data: pd.DataFrame
    estimated_orientations_: list
    sampling_rate_hz: float
    initial_orientation: Rotation

    def __init__(self, initial_orientation: Rotation):
        self.initial_orientation = initial_orientation

    def estimate(self, sensor_data: pd.DataFrame, sampling_rate_hz: float):
        """Use the initial rotation and the gyroscope signal to estimate rotations of second until last sample.

        Parameters
        ----------
        sensor_data
            At least must contain 3D-gyroscope data for one sensor (multiple sensors not yet implemented).
        sampling_rate_hz
            Sampling rate with which gyroscopic data was recorded.

        Notes
        -----
        This function makes use of :func:`scipy.spatial.transform.Rotation.from_rotvec`, which updates a quaternion
        by using norm of gyroscopic data as amplitude of rotation and the normalized vector of gyroscopic data for
        axis of rotation.

        Examples
        --------
        >>> gyr_integrator.estimate(sensor_data, 204.8)
        >>> orientations = gyr_integrator.estimated_orientations_
        >>> orientations[-1].as_quat()
        array([0., 1, 0., 0.])

        """
        # TODO: so far it is only possible to pass one sensor with columns being gyr_x...(and possilby others acc_z)
        #  ==> adapt to multiple sensors
        self.sampling_rate_hz = sampling_rate_hz
        self.estimated_orientations_ = []
        self.estimated_orientations_.append(self.initial_orientation)
        for i_sample in range(1, len(sensor_data) + 1):
            previous_quat = self.estimated_orientations_[i_sample - 1]
            update_quat = Rotation.from_rotvec(sensor_data[SF_GYR].iloc[i_sample - 1] / self.sampling_rate_hz)
            self.estimated_orientations_.append(previous_quat * update_quat)
        return self
