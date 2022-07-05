"""The :py:mod:`gaitmap.event_detection` contains algorithms to detect temporal gait events in the sensor signal.

Different algorithms for event detection are going to be collected here.
"""

from gaitmap_mad.event_detection._rampp_event_detection import RamppEventDetection
from gaitmap_mad.event_detection._rampp_event_detection_withfilter import RamppEventDetectionFilter

__all__ = ["RamppEventDetection", "RamppEventDetectionFilter"]
