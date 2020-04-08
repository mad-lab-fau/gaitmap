"""Estimation of velocity and position relative to first sample of passed data."""
import numpy as np
import numpy.matlib
import pandas as pd
from scipy import integrate
from typing import Optional

from gaitmap.base import BasePositionEstimation
from gaitmap.utils import dataset_helper
from gaitmap.utils.consts import SF_ACC, SF_VEL, SF_POS


class ForwardBackwardIntegration(BasePositionEstimation):
    """Use forward integration of acceleration to estimate velocity and position.

    For drift removal, backward integration is used for velocity estimation, because we assume zero velocity at the
    beginning and end of a signal. For position, drift removal via backward integration is only used for the vertical
    axis (=z-axis or superior-inferior-axis, see :ref:`ff`, because we assume beginning and end of the motion are in
    one plane. Implementation based on the paper by Hannink et al. [1]

    Attributes
    ----------
    velocity_
        The velocity estimated by direct-and-reverse / forward-backward integration.
    position_
        The position estimated by forward integration in the ground plane and by direct-and-reverse /
        forward-backward integration for the vertical axis.


    Parameters
    ----------
    turning_point
        The point at which the sigmoid weighting function has a value of 0.5 and therefore forward and backward
        integrals are weighted 50/50. Specified as percentage of the signal length (0.0 < turning_point <= 1.0)

    steepness
        Steepness of the sigmoid function to weight forward and backward integral.

    Other Parameters
    ----------------
    data
        The data passed to the `estimate` method.
    sampling_rate_hz
        The sampling rate of the data.

    Notes
    -----
    TODO: add support for multiple sensors and adapt tests accordingly
    .. [1] Hannink, J., Ollenschläger, M., Kluge, F., Roth, N., Klucken, J., and Eskofier, B. M. 2017. Benchmarking Foot
       Trajectory Estimation Methods for Mobile Gait Analysis. Sensors (Basel, Switzerland) 17, 9.
       https://doi.org/10.3390/s17091940

    Examples
    --------
    >>> spatial = ForwardBackwardIntegration(0.5, 0.08)
    >>> spatial.estimate(data, 204.8)
    >>> spatial.velocity_.iloc[-1]
    vel_x   -1.175808e-15
    vel_y   -1.175836e-15
    vel_z   -1.175865e-15
    Name: 1999, dtype: float64

    """

    sampling_rate_hz: float
    steepness: float
    turning_point: float
    velocity_: pd.DataFrame
    position_: pd.DataFrame

    def __init__(self, turning_point: Optional[float] = 0.5, steepness: Optional[float] = 0.08):
        self.turning_point = turning_point
        self.steepness = steepness

    def estimate(self, data, sampling_rate_hz):
        """Estimate velocity and position based on acceleration data."""
        if not (0.0 <= self.turning_point <= 1.0):
            raise ValueError(
                "Bad ForwardBackwardIntegration initialization found. Turning point must be in the rage "
                "of 0.0 to 1.0"
            )

        if dataset_helper.is_multi_sensor_dataset(data):
            raise NotImplementedError("Multisensor input is not supported yet")

        if not dataset_helper.is_single_sensor_dataset(data):
            raise ValueError("Provided data set is not supported by gaitmap")

        self.position_ = pd.DataFrame(columns=SF_POS, index=data.index)
        self.sampling_rate_hz = sampling_rate_hz

        self.velocity_ = self._forward_backward_integration(data, SF_ACC)
        self.velocity_.columns = SF_VEL
        self.position_[SF_POS[2]] = self._forward_backward_integration(self.velocity_, SF_VEL[2])
        self.position_[SF_POS[1]] = (
            integrate.cumtrapz(self.velocity_[SF_VEL[1]], axis=0, initial=0) / self.sampling_rate_hz
        )
        self.position_[SF_POS[0]] = (
            integrate.cumtrapz(self.velocity_[SF_VEL[0]], axis=0, initial=0) / self.sampling_rate_hz
        )
        self.position_.columns = SF_POS
        # TODO: implement integration of velocity to obtain position
        return self

    def _get_weight_matrix(self, data_to_integrate: pd.DataFrame):
        n_samples = data_to_integrate.shape[0]
        n_axes = data_to_integrate.shape[1]

        x = np.linspace(0, 1, n_samples)
        s = 1 / (1 + np.exp(-(x - self.turning_point) / self.steepness))
        weights = (s - s[0]) / (s[-1] - s[0])

        return np.tile(weights, n_axes).reshape(n_samples, n_axes)

    def _forward_backward_integration(self, data, channels):
        # TODO: make it possible to set initial value of integral from outside?
        integral_forward = integrate.cumtrapz(data[channels], axis=0, initial=0) / self.sampling_rate_hz
        integral_backward = integrate.cumtrapz(np.flipud(data[channels]), axis=0, initial=0) / self.sampling_rate_hz
        weights_vel = self._get_weight_matrix(pd.DataFrame(data[channels]))

        return pd.DataFrame(integral_forward * weights_vel + integral_backward * (1 - weights_vel), index=data.index)
