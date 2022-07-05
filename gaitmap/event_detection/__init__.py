"""The :py:mod:`gaitmap.event_detection` contains algorithms to detect temporal gait events in the sensor signal.

Different algorithms for event detection are going to be collected here.
"""

from gaitmap.event_detection._herzer_event_detection import HerzerEventDetection
from gaitmap.utils._gaitmap_mad import patch_gaitmap_mad_import

_gaitmap_mad_modules = {"RamppEventDetection", "FilteredRamppEventDetection"}

if not (__getattr__ := patch_gaitmap_mad_import(_gaitmap_mad_modules, __name__)):
    del __getattr__
    from gaitmap_mad.event_detection import RamppEventDetection, FilteredRamppEventDetection

__all__ = ["RamppEventDetection", "HerzerEventDetection", "FilteredRamppEventDetection"]
