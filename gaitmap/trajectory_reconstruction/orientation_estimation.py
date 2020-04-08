"""Estimation of orientations by gyroscope integration."""
import operator
from itertools import accumulate

import pandas as pd
from scipy.spatial.transform import Rotation
from typing import Union, Dict

from gaitmap.base import BaseOrientationEstimation
from gaitmap.utils import dataset_helper
from gaitmap.utils.consts import SF_GYR
from gaitmap.utils.dataset_helper import SingleSensorDataset, MultiSensorDataset, get_multi_sensor_dataset_names
from gaitmap.utils.dataset_helper import is_single_sensor_dataset, is_multi_sensor_dataset


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

    data: Union[SingleSensorDataset, MultiSensorDataset]
    sampling_rate_hz: float

    def __init__(self, initial_orientation: Union[Rotation, Dict[str, Rotation]]):
        self.initial_orientation = initial_orientation

    def estimate(self, data: Union[SingleSensorDataset, MultiSensorDataset], sampling_rate_hz: float):
        """Use the initial rotation and the gyroscope signal to estimate the orientation to every time point .

        Parameters
        ----------
        data
            At least must contain 3D-gyroscope data.
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

        if is_single_sensor_dataset(self.data):
            self
            self.estimated_orientations_, self.estimated_orientations_with_initial_ = self._estimate_single_sensor(
                self.data
            )
        elif is_multi_sensor_dataset(self.data):
            (self.estimated_orientations_, self.estimated_orientations_with_initial_,) = self._estimate_multi_sensor(
                data
            )
        else:
            raise ValueError("Provided data set is not supported by gaitmap")
        return self

    def _estimate_single_sensor(self, data) -> Rotation:
        gyro_data = data[SF_GYR].to_numpy()
        single_step_rotations = Rotation.from_rotvec(gyro_data / self.sampling_rate_hz)
        # This is faster than np.cumprod. Custom quat rotation would be even faster, as we could skip the second loop
        out = accumulate([self.initial_orientation, *single_step_rotations], operator.mul)
        out_as_rot = Rotation([o.as_quat() for o in out])
        return out_as_rot[1:], out_as_rot

    def _estimate_multi_sensor(self, data: MultiSensorDataset) -> Rotation:
        if isinstance(data, dict):
            for i_sensor in data:
                (
                    self.estimated_orientations_[i_sensor],
                    self.estimated_orientations_with_initial_[i_sensor],
                ) = self._estimate_single_sensor(self.data[i_sensor])
        elif isinstance(data, pd.DataFrame):
            sensors = get_multi_sensor_dataset_names(data)
            self._estimated_orientations_ = {k: 0 for k in sensors}
            self._estimated_orientations_with_initial_ = {k: 0 for k in sensors}
            for i_sensor in sensors:
                (
                    self.estimated_orientations_[i_sensor],
                    self.estimated_orientations_with_initial_[i_sensor],
                ) = self._estimate_single_sensor(self.data.xs(i_sensor, level=0, axis=1))
        else:
            raise ValueError("Given format of multisensor not supported. See `utils.dataset_helper` for supported "
                             "types")
