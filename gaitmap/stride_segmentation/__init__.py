"""The :py:mod:`gaitmap.stride_segmentation` contains popular algorithms to detect strides in a sensor signal.

The stride segmentation module includes all algorithms that are able to find stride candidates in a continuous sensor
signal.
Some are able to directly detect individual biomechanical events.
Other algorithm are only able to detect stride candidates and need to be paired by an explicit event detection
algorithm, as implemented in :py:mod:`gaitmap.event_detection`, to be able to provide information about biomechanical
events.
"""

from gaitmap.stride_segmentation.barth_dtw import BarthDtw
from gaitmap.stride_segmentation.base_dtw import BaseDtw
from gaitmap.stride_segmentation.dtw_templates import DtwTemplate, create_dtw_template


__all__ = ["BarthDtw", "BaseDtw", "DtwTemplate", "create_dtw_template"]