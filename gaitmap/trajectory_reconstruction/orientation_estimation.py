"""Estimation of orientations by trapezoidal gyroscope integration"""
from gaitmap.base import BaseOrientationEstimation
from numpy import cumsum
from scipy.spatial.transform import Rotation
from gaitmap.utils.consts import SF_GYR, SF_ACC
from pandas import DataFrame


class GyroIntegration(BaseOrientationEstimation):
    """Estimate orientation based on a given initial orientation. Subsequent orientations are estimated by
    integration gyroscope data with respect to time using the trapezoidal rule.



    Parameters
    ----------
    sensor_data : pandas.DataFrame
        contains gyroscope and acceleration data
    estimated_orientations : scipy.spatial.transform.Rotation
        contains rotations based on initial rotation and trapezoidal gyroscope integration

    """

    sensor_data: DataFrame
    estimated_orientations: DataFrame

    def __init__(self, sensor_data):
        self.sensor_data = sensor_data  # .copy()?

    def estimate_orientation_sequence(self, initial_orientation: Rotation, sensor_data):
        gyr = sensor_data[SF_GYR]
        acc = sensor_data[SF_ACC]

