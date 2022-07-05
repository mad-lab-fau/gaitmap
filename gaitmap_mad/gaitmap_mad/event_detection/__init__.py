"""The :py:mod:`gaitmap.event_detection` contains algorithms to detect temporal gait events in the sensor signal.

Different algorithms for event detection are going to be collected here.
"""
from gaitmap_mad.event_detection._filtered_rampp_event_detection import FilteredRamppEventDetection
from gaitmap_mad.event_detection._rampp_event_detection import RamppEventDetection

__all__ = ["RamppEventDetection", "FilteredRamppEventDetection"]
