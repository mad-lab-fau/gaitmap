"""Estimation of velocity and position relative to first sample of passed data."""
import numpy as np
import numpy.matlib
import pandas as pd
from scipy import integrate

from gaitmap.base import BasePositionEstimation
from gaitmap.utils import dataset_helper
from gaitmap.utils.consts import SF_ACC


class ForwardBackwardIntegration(BasePositionEstimation):
    """Use forward integration of acceleration to estimate velocity and position.

    For drift removal, backward integration is used for velocity estimation, because we assume zero velocity at the
    beginning and end of a signal. For position, drift removal via backward integration is only used for the vertical
    axis (=z-axis or superior-inferior-axis, see :ref:`ff`, because we assume beginning and end of the motion are in
    one plane. Implementation based on the paper by Hannink et al. [1]

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

    """

    sampling_rate_hz: float
    # TODO: @Reviewwer: the next two lines don't need to be in "self", right? Put them somewhere else? consts?
    vel_axis_names = [i_axis.replace("acc", "vel") for i_axis in SF_ACC]
    pos_axis_names = [i_axis.replace("acc", "vel") for i_axis in SF_ACC]

    velocity_: pd.DataFrame
    position_: pd.DataFrame
    steepness: float
    turning_point: float

    # TODO: this was taken from Julius' Benchmarking paper, adapt to default value but make it possible to pass a
    #  different x0 to estimate
    x0 = 0.6  # TODO: make this optional

    def __init__(self, turning_point, steepness):
        # if turning_point < 0 or turning_point > 1.0:
        if not (0.0 <= turning_point <= 1.0):
            raise ValueError("Turning point must be in the rage of 0.0 to 1.0")
        self.turning_point = turning_point
        self.steepness = steepness

    def estimate(self, data, sampling_rate_hz):
        """Estimate velocity and position based on acceleration data."""
        if dataset_helper.is_multi_sensor_dataset(data):
            raise NotImplementedError("Multisensor input is not supported yet")

        if not dataset_helper.is_single_sensor_dataset(data):
            raise ValueError("Provided data set is not supported by gaitmap")

        # TODO: is this way of using vel_axis_names OK or should it be ForwardBackwardIntegration.vel...?
        self.position_ = pd.DataFrame(columns=self.pos_axis_names, index=data.index)
        self.sampling_rate_hz = sampling_rate_hz

        self.velocity_ = self._forward_backward_integration(data, SF_ACC)
        self.velocity_.columns = self.vel_axis_names
        self.position_[self.pos_axis_names[2]] = self._forward_backward_integration(
            self.velocity_, self.vel_axis_names[2]
        )
        self.position_[self.pos_axis_names[1]] = (
            integrate.cumtrapz(self.velocity_[self.vel_axis_names[1]], axis=0, initial=0) / self.sampling_rate_hz
        )
        self.position_[self.pos_axis_names[0]] = (
            integrate.cumtrapz(self.velocity_[self.vel_axis_names[1]], axis=0, initial=0) / self.sampling_rate_hz
        )
        self.position_.columns = self.pos_axis_names
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
