"""Estimation of velocity and position relative to first sample of passed data."""
import numpy as np
import pandas as pd

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
    # TODO: change name of volumns to 'vel_axis' and 'pos_axis'

    velocity = pd.DataFrame(SF_ACC)
    position = pd.DataFrame(SF_ACC)

    def __init__(self, parameter):
        self.parameter = parameter

    def estimate(self, data):
        integral_forward = np.cumtrapz(data[SF_ACC]) / self.sampling_rate_hz
        integral_backward = np.cumtrapz(np.flipud(data[SF_ACC])) / self.sampling_rate_hz
        pass

    def _get_weighting_function(self,):
        pass
