"""Estimation of velocity and position relative to first sample of passed data."""
import numpy as np
import numpy.matlib
import pandas as pd
from scipy import integrate

from gaitmap.base import BasePositionEstimation
from gaitmap.utils.consts import SF_ACC


class ForwardBackwardIntegration(BasePositionEstimation):
    """Use forward integration of acceleration to estimate velocity and position.

    For drift removal, backward integration is used for velocity estimation, because we assume zero velocity at the
    beginning and end of a signal. For position, drift removal via backward integration is only used for the vertical
    axis (=z-axis or superior-inferior-axis, see :ref:`ff`.
    """

    # TODO: add support for multiple sensors
    sampling_rate_hz: float
    # TODO: add support for body frame axes (medical axes), e.g. using
    #  velocity.columns.map({'acc_ml':'vel_ml',...})

    # TODO: the next two lines don't need to be in "self", right? Put them some

    vel_axis_names = [i_axis.replace("acc", "vel") for i_axis in SF_ACC]
    pos_axis_names = [i_axis.replace("acc", "vel") for i_axis in SF_ACC]

    velocity: pd.DataFrame
    position: pd.DataFrame

    # TODO: this was taken from Julius' Benchmarking paper, adapt to default value but make it possible to pass a
    #  different x0 to estimate
    x0 = 0.6  # TODO: make this optional

    def __init__(self, x0):
        self.x0 = x0

    def estimate(self, data):
        # TODO: is this way of using vel_axis_names OK?
        self.velocity = pd.DataFrame(index=ForwardBackwardIntegration.vel_axis_names)
        integral_forward = integrate.cumtrapz(data[SF_ACC], axis=0) / self.sampling_rate_hz
        integral_backward = integrate.cumtrapz(np.flipud(data[SF_ACC]), axis=0) / self.sampling_rate_hz

        weights_vel = self._get_weight_matrix(data[SF_ACC])
        self.velocity = pd.DataFrame(
            integral_forward * weights_vel + integral_backward * (1 - weights_vel),
            columns=ForwardBackwardIntegration.vel_axis_names,
            index=data.index,
        )

        # TODO: implement integration of velocity to obtain position
        return self

    def _get_weight_matrix(self, data_to_integrate):
        x = np.linspace(0, 1, len(data_to_integrate))
        s = 1 / (1 + np.exp(-(x - self.x0)))
        weights = (s - s[0]) / (s[-1] - s[0])

        return np.tile(weights, 3).reshape(len(data_to_integrate), 3)
