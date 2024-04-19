"""Methods to calculate the global orientation and position of an IMU.

This module provides simple methods to estimate the orientation and position on custom IMU data and a set of wrappers
to make applying these methods to the default gaitmap datasets easier.
"""

from gaitmap.trajectory_reconstruction._region_level_trajectory import RegionLevelTrajectory
from gaitmap.trajectory_reconstruction._stride_level_trajectory import StrideLevelTrajectory
from gaitmap.trajectory_reconstruction.orientation_methods import MadgwickAHRS, SimpleGyroIntegration
from gaitmap.trajectory_reconstruction.position_methods._forward_backwards_integration import ForwardBackwardIntegration
from gaitmap.trajectory_reconstruction.trajectory_methods import MadgwickRtsKalman, RtsKalman
from gaitmap.utils._gaitmap_mad import patch_gaitmap_mad_import

_gaitmap_mad_modules = {
    "PieceWiseLinearDedriftedIntegration",
}

if not (__getattr__ := patch_gaitmap_mad_import(_gaitmap_mad_modules, __name__)):
    del __getattr__
    from gaitmap_mad.trajectory_reconstruction.position_methods import PieceWiseLinearDedriftedIntegration

__all__ = [
    "MadgwickAHRS",
    "SimpleGyroIntegration",
    "ForwardBackwardIntegration",
    "RtsKalman",
    "MadgwickRtsKalman",
    "StrideLevelTrajectory",
    "RegionLevelTrajectory",
    "PieceWiseLinearDedriftedIntegration",
]
