"""The :py:mod:`gaitmap.gait_detection` contains algorithms to detect regions of gait in the sensor signal.

Different algorithms for gait sequence detection are going to be collected here.
"""

from gaitmap_mad.gait_detection._ullrich_gait_sequence_detection import UllrichGaitSequenceDetection

__all__ = ["UllrichGaitSequenceDetection"]
