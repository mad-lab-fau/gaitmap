"""Methods to calculate the global orientation and position of an IMU.

This module provides simple methods to estimate the orientation and position on custom IMU data and a set of wrappers
to make applying these methods to the default gaitmap datasets easier.
"""

from gaitmap.trajectory_reconstruction.orientation_methods import MadgwickAHRS, SimpleGyroIntegration
from gaitmap.trajectory_reconstruction.position_methods import (
    ForwardBackwardIntegration,
    PieceWiseLinearDedriftedIntegration,
)
from gaitmap.trajectory_reconstruction.region_level_trajectory import RegionLevelTrajectory
from gaitmap.trajectory_reconstruction.stride_level_trajectory import StrideLevelTrajectory
from gaitmap.trajectory_reconstruction.trajectory_methods import RtsKalman

__all__ = [
    "MadgwickAHRS",
    "SimpleGyroIntegration",
    "ForwardBackwardIntegration",
    "RtsKalman",
    "StrideLevelTrajectory",
    "RegionLevelTrajectory",
    "PieceWiseLinearDedriftedIntegration",
]
