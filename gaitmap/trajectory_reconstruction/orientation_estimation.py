from gaitmap.base import BaseOrientationEstimation
from scipy.spatial.transform import Rotation


class GyroIntegration(BaseOrientationEstimation):
    def __init__(self):
        pass

    def estimate_orientation_sequence(self, initial_orientation: Rotation, sensor_data):
        pass
