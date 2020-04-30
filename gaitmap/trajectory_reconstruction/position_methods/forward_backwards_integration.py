from typing import Optional

import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz

from gaitmap.base import BasePositionMethod, BaseType
from gaitmap.utils.consts import GRAV_VEC, SF_ACC, GF_VEL, GF_POS
from gaitmap.utils.dataset_helper import SingleSensorDataset, is_single_sensor_dataset


class ForwardBackwardIntegration(BasePositionMethod):
    """Use forward(-backward) integration of acc to estimate velocity and position.

    .. warning::
       We assume that the acc signal is already converted into the world frame!

    This method uses the zero-velocity assumption (ZUPT) to perform a drift removal using a direct-and-reverse (DRI) or
    forward-backward integration.
    This means we assume zero velocity at the beginning and end of the signal.

    Further drift correction is applied using a level-assumption (i.e. we assume that the sensor starts and ends its
    movement at the same z-position/height).
    If this assumption is not true for your usecase, you can disable it using the `level_assumption` parameter.

    Implementation based on the paper by Hannink et al. [1]_.

    TODO: Add info to docs about what to do to only use forward or only backwards.

    Notes
    -----
    .. [1] Hannink, J., Ollenschläger, M., Kluge, F., Roth, N., Klucken, J., and Eskofier, B. M. 2017. Benchmarking Foot
       Trajectory Estimation Methods for Mobile Gait Analysis. Sensors (Basel, Switzerland) 17, 9.
       https://doi.org/10.3390/s17091940
    """

    steepness: float
    turning_point: float
    level_assumption: bool
    gravity: Optional[np.ndarray]

    data: SingleSensorDataset
    sampling_rate_hz: float

    def __init__(
        self,
        turning_point: float = 0.5,
        steepness: float = 0.08,
        level_assumption: bool = True,
        gravity: Optional[np.ndarray] = GRAV_VEC,
    ):
        self.turning_point = turning_point
        self.steepness = steepness
        self.level_assumption = level_assumption
        self.gravity = gravity

    def estimate(self: BaseType, data: SingleSensorDataset, sampling_rate_hz: float) -> BaseType:
        if not 0.0 <= self.turning_point <= 1.0:
            raise ValueError("`turning_point` must be in the rage of 0.0 to 1.0")
        if not is_single_sensor_dataset(data, check_gyr=False, frame="sensor"):
            raise ValueError("Data is not a single sensor dataset.")
        self.sampling_rate_hz = sampling_rate_hz
        self.data = data

        acc_data = data[SF_ACC].to_numpy()
        if self.gravity is not None:
            acc_data -= self.gravity

        velocity = self._forward_backward_integration(acc_data)
        position_xy = cumtrapz(velocity[:, :2], axis=0, initial=0) / self.sampling_rate_hz
        if self.level_assumption is True:
            position_z = self._forward_backward_integration(velocity[:, [2]])
        else:
            position_z = cumtrapz(velocity[:, [2]], axis=0, initial=0) / self.sampling_rate_hz
        position = np.hstack((position_xy, position_z))

        self.velocity_ = pd.DataFrame(velocity, columns=GF_VEL)
        self.position_ = pd.DataFrame(position, columns=GF_POS)

        return self

    def _sigmoid_weight_function(self, n_samples: int) -> np.ndarray:
        x = np.linspace(0, 1, n_samples)
        s = 1 / (1 + np.exp(-(x - self.turning_point) / self.steepness))
        weights = (s - s[0]) / (s[-1] - s[0])
        return weights

    def _forward_backward_integration(self, data: np.ndarray) -> np.ndarray:
        # TODO: different steepness and turning point for velocity and position?
        integral_forward = cumtrapz(data, axis=0, initial=0) / self.sampling_rate_hz
        # for backward integration, we flip the signal and inverse the time by using a negative sampling rate.
        integral_backward = cumtrapz(data[::-1], axis=0, initial=0) / -self.sampling_rate_hz
        weights = self._sigmoid_weight_function(data.shape[0])

        return (integral_forward.T * (1 - weights) + integral_backward[::-1].T * weights).T
