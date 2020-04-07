"""Estimation of orientations by gyroscope integration."""
import operator
from itertools import accumulate

from scipy.spatial.transform import Rotation

from gaitmap.base import BaseOrientationEstimation
from gaitmap.utils import dataset_helper
from gaitmap.utils.consts import SF_GYR
from gaitmap.utils.dataset_helper import SingleSensorDataset


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
    >>> gyr_integrator = GyroIntegration(Rotation([0, 0, 1, 0]))

    """

    initial_orientation: Rotation

    estimated_orientations_: Rotation
    estimated_orientations_with_initial_: Rotation

    data: SingleSensorDataset
    sampling_rate_hz: float

    def __init__(self, initial_orientation: Rotation):
        self.initial_orientation = initial_orientation

    def estimate(self, data: SingleSensorDataset, sampling_rate_hz: float):
        """Use the initial rotation and the gyroscope signal to estimate the orientation to every time point .

        Parameters
        ----------
        data
            At least must contain 3D-gyroscope data for one sensor (multiple sensors not yet implemented).
        sampling_rate_hz
            Sampling rate with which gyroscopic data was recorded.

        Notes
        -----
        This function makes use of `from_rotvec` of :func:`~scipy.spatial.transform.Rotation`, to turn the gyro signal
        of each sample into a differential quaternion.
        This means that the rotation between two samples is assumed to be constant around one axis.

        Examples
        --------
        >>> gyr_integrator = GyroIntegration()
        >>> gyr_integrator.estimate(data, 204.8)
        >>> orientations = gyr_integrator.estimated_orientations_
        >>> orientations[-1].as_quat()
        array([0., 1, 0., 0.])

        """
        # TODO: so far it is only possible to pass one sensor with columns being gyr_x...(and possilby others acc_z)
        #  ==> adapt to multiple sensors

        if dataset_helper.is_multi_sensor_dataset(data):
            raise NotImplementedError("Multisensor input is not supported yet")

        if not dataset_helper.is_single_sensor_dataset(data):
            raise ValueError("Provided data set is not supported by gaitmap")

        self.sampling_rate_hz = sampling_rate_hz
        gyro_data = data[SF_GYR].to_numpy()
        single_step_rotations = Rotation.from_rotvec(gyro_data / sampling_rate_hz)
        # This is faster than np.cumprod. Custom quat rotation would be even faster, as we could skip the second loop
        out = accumulate([self.initial_orientation, *single_step_rotations], operator.mul)
        # Exclude initial orientation
        out_as_rot = Rotation([o.as_quat() for o in out])
        self.estimated_orientations_ = out_as_rot[1:]
        self.estimated_orientations_with_initial_ = out_as_rot
        return self
