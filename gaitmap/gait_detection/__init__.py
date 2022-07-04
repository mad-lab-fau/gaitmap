"""The :py:mod:`gaitmap.gait_detection` contains algorithms to detect regions of gait in the sensor signal.

Different algorithms for gait sequence detection are going to be collected here.
"""
from gaitmap.utils._gaitmap_mad import patch_gaitmap_mad_import

_gaitmap_mad_modules = {
    "UllrichGaitSequenceDetection",
}

if not (__getattr__ := patch_gaitmap_mad_import(_gaitmap_mad_modules, __name__)):
    del __getattr__
    from gaitmap_mad.gait_detection import UllrichGaitSequenceDetection


__all__ = ["UllrichGaitSequenceDetection"]
