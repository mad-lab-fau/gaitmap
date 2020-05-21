"""The :py:mod:`gaitmap.event_detection` contains algorithms to detect temporal gait events in the sensor signal.

Different algorithms for event detection are going to be collected here.
"""

from gaitmap.event_detection.rampp_event_detection import RamppEventDetection
from gaitmap.event_detection.stride_list_conversions import enforce_stride_list_consistency

__all__ = ["RamppEventDetection"]
