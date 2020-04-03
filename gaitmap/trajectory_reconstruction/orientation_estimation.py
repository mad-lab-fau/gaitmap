"""Estimation of orientations by trapezoidal gyroscope integration."""
from pandas import DataFrame
from scipy.spatial.transform import Rotation

from gaitmap.base import BaseOrientationEstimation


class GyroIntegration(BaseOrientationEstimation):
    """Estimate orientation based on a given initial orientation.

    Subsequent orientations are estimated by integration gyroscope data with respect to time using the trapezoidal rule.

    Parameters
    ----------
    sensor_data : pandas.DataFrame
        contains gyroscope and acceleration data

    Attributes
    ----------
    estimated_orientations : scipy.spatial.transform.Rotation
        contains rotations based on initial rotation and trapezoidal gyroscope integration

    """

    sensor_data: DataFrame
    estimated_orientations: DataFrame

    def __init__(self, sensor_data):
        self.sensor_data = sensor_data  # .copy()?

    def estimate_orientation_sequence(self, initial_orientation: Rotation, sensor_data):
        pass
        # gyr = sensor_data[SF_GYR]
