"""The :py:mod:`gaitmap.stride_segmentation` contains popular algorithms to detect strides in a sensor signal.

The stride segmentation module includes all algorithms that are able to find stride candidates in a continuous sensor
signal.
Some are able to directly detect individual biomechanical events.
Other algorithm are only able to detect stride candidates and need to be paired by an explicit event detection
algorithm, as implemented in :py:mod:`gaitmap.event_detection`, to be able to provide information about biomechanical
events.
"""
from gaitmap_mad.stride_segmentation.dtw import (
    BarthDtw,
    BarthOriginalTemplate,
    BaseDtw,
    BaseDtwTemplate,
    ConstrainedBarthDtw,
    DtwTemplate,
    InterpolatedDtwTemplate,
    TrainableTemplateMixin,
    find_matches_find_peaks,
    find_matches_min_under_threshold,
)

__all__ = [
    "BaseDtw",
    "BarthDtw",
    "ConstrainedBarthDtw",
    "find_matches_find_peaks",
    "find_matches_min_under_threshold",
    "BarthOriginalTemplate",
    "BaseDtwTemplate",
    "DtwTemplate",
    "InterpolatedDtwTemplate",
    "TrainableTemplateMixin",
]
