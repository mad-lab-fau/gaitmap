"""Naive Integration of Gyroscope to estimate the orientation."""
from typing import Union

import numpy as np
from numba import njit
from scipy.spatial.transform import Rotation

from gaitmap.base import BaseType, BaseOrientationMethod
from gaitmap.utils.consts import SF_GYR
from gaitmap.utils.dataset_helper import SingleSensorDataset, is_single_sensor_dataset
from gaitmap.utils.fast_quaternion_math import rate_of_change_from_gyro


class SimpleGyroIntegration(BaseOrientationMethod):
    """Integrate Gyro values without any drift correction.

    Parameters
    ----------
    initial_orientation : Rotation or (1 x 4) quaternion array
        The initial orientation used.
        If you pass a array, remember that the order of elements must be x, y, z, w.

    Attributes
    ----------
    orientations_
        The final array of rotations **including** the initial orientation.
    orientation_list_
        The rotations as a *SingleSensorOrientationList*

    Other Parameters
    ----------------
    data
        The data passed to the estimate method
    sampling_rate_hz
        The sampling rate of this data

    """

    initial_orientation: Union[np.ndarray, Rotation]

    orientation_: Rotation

    data: SingleSensorDataset
    sampling_rate_hz: float

    def __init__(self, initial_orientation: Union[np.ndarray, Rotation] = Rotation.identity()):
        self.initial_orientation = initial_orientation

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
        if not is_single_sensor_dataset(data, check_acc=False, frame="sensor"):
            raise ValueError("Data is not a single sensor dataset.")
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        initial_orientation = self.initial_orientation
        if isinstance(initial_orientation, Rotation):
            initial_orientation = Rotation.as_quat(initial_orientation)
        gyro_data = np.deg2rad(data[SF_GYR].to_numpy())

        rots = _simple_gyro_integration_series(
            gyro=gyro_data, initial_orientation=initial_orientation, sampling_rate_hz=sampling_rate_hz,
        )
        self.orientation_rot_ = Rotation.from_quat(rots)

        return self


@njit()
def _simple_gyro_integration_series(gyro, initial_orientation, sampling_rate_hz):
    out = np.empty((len(gyro) + 1, 4))
    q = initial_orientation
    out[0] = q
    for i in range(len(gyro)):
        qdot = rate_of_change_from_gyro(gyro[i], q)
        # Integrate rate of change of quaternion to yield quaternion
        q += qdot / sampling_rate_hz
        q /= np.sqrt(np.sum(q ** 2))
        out[i + 1] = q

    return out
