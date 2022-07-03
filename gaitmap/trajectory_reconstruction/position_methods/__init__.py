"""Methods to calculate the global position of an IMU."""

from gaitmap.trajectory_reconstruction.position_methods._forward_backwards_integration import ForwardBackwardIntegration
from gaitmap.utils._gaitmap_mad import patch_gaitmap_mad_import

_gaitmap_mad_modules = {
    "PieceWiseLinearDedriftedIntegration",
}

if not (__getattr__ := patch_gaitmap_mad_import(_gaitmap_mad_modules, __name__)):
    from gaitmap_mad.trajectory_reconstruction.position_methods import PieceWiseLinearDedriftedIntegration


__all__ = [
    "ForwardBackwardIntegration",
    "PieceWiseLinearDedriftedIntegration",
]
