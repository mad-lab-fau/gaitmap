"""Naive Integration of Gyroscope to estimate the orientation."""
import operator
from itertools import accumulate
from typing import Union

import numpy as np
from scipy.spatial.transform import Rotation

from gaitmap.base import BaseType, BaseOrientationMethods
from gaitmap.utils.consts import SF_GYR
from gaitmap.utils.dataset_helper import SingleSensorDataset, is_single_sensor_dataset


class SimpleGyroIntegration(BaseOrientationMethods):
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

    orientations_: Rotation

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
        if isinstance(initial_orientation, np.ndarray):
            initial_orientation = Rotation.from_quat(initial_orientation)
        gyro_data = np.deg2rad(data[SF_GYR].to_numpy())
        single_step_rotations = Rotation.from_rotvec(gyro_data / self.sampling_rate_hz)
        # This is faster than np.cumprod. Custom quat rotation would be even faster, as we could skip the second loop
        out = accumulate([initial_orientation, *single_step_rotations], operator.mul)
        self.orientations_ = Rotation([o.as_quat() for o in out])
        return self
