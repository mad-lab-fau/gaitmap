"""The :py:mod:`gaitmap.event_detection` contains algorithms to detect temporal gait events in the sensor signal.

Different algorithms for event detection are going to be collected here.
"""

from gaitmap.event_detection.rampp_event_detection import RamppEventDetection

__all__ = ["RamppEventDetection"]
