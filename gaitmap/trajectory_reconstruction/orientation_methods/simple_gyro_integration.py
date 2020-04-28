import operator
from itertools import accumulate
from typing import Union

import numpy as np
from scipy.spatial.transform import Rotation

from gaitmap.base import BaseType, BaseOrientationMethods
from gaitmap.utils.consts import SF_GYR
from gaitmap.utils.dataset_helper import SingleSensorDataset


class SimpleGyroIntegration(BaseOrientationMethods):

    initial_orientation: Union[np.ndarray, Rotation]

    orientations_: Rotation

    data: SingleSensorDataset
    sampling_rate_hz: float

    def __init__(self, initial_orientation: Union[np.ndarray, Rotation] = Rotation.identity()):
        self.initial_orientation = initial_orientation

    # TODO: Allow to continue the integration
    def estimate(self: BaseType, data: SingleSensorDataset, sampling_rate_hz: float) -> BaseType:
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        initial_orientation = self.initial_orientation
        if isinstance(initial_orientation, np.ndarray):
            initial_orientation = Rotation.from_quat(initial_orientation)
        gyro_data = data[SF_GYR].to_numpy()
        single_step_rotations = Rotation.from_rotvec(np.deg2rad(gyro_data) / self.sampling_rate_hz)
        # This is faster than np.cumprod. Custom quat rotation would be even faster, as we could skip the second loop
        out = accumulate([initial_orientation, *single_step_rotations], operator.mul)
        self.orientations_ = Rotation([o.as_quat() for o in out])
        return self
