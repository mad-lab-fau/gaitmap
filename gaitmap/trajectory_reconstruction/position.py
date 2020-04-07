"""Estimation of velocity and position relative to first sample of passed data."""
from gaitmap.base import BasePositionEstimation


class ForwardBackwardIntegration(BasePositionEstimation):
    """Use forward integration of acceleration to estimate velocity and position.

    For drift removal, backward integration is used for velocity estimation, because we assume zero velocity at the
    beginning and end of a signal. For position, drift removal via backward integration is only used for the vertical
    axis (=z-axis or superior-inferior-axis, see :ref:`ff`.
    """

    def __init__(self, parameter):
        self.parameter = parameter

    def estimate(self, data):
        pass
