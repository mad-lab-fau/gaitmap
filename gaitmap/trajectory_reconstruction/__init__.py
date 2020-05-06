"""Methods to calculate the global orientation and position of an IMU.

This module provides simple methods to estimate the orientation and position on custom IMU data and a set of wrappers
to make applying these methods to the default gaitmap datasets easier.
"""

from gaitmap.trajectory_reconstruction.orientation_methods import MadgwickAHRS, SimpleGyroIntegration
from gaitmap.trajectory_reconstruction.position_methods import ForwardBackwardIntegration
from gaitmap.trajectory_reconstruction.stride_level_trajectory import StrideLevelTrajectory

__all__ = ["MadgwickAHRS", "SimpleGyroIntegration", "ForwardBackwardIntegration", "StrideLevelTrajectory"]
