"""Methods to calculate the global orientation of an IMU."""

from gaitmap.trajectory_reconstruction.orientation_methods._madgwick import MadgwickAHRS
from gaitmap.trajectory_reconstruction.orientation_methods._simple_gyro_integration import SimpleGyroIntegration

__all__ = ["MadgwickAHRS", "SimpleGyroIntegration"]
