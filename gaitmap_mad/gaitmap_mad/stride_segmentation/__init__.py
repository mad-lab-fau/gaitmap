"""The :py:mod:`gaitmap.stride_segmentation` contains popular algorithms to detect strides in a sensor signal.

The stride segmentation module includes all algorithms that are able to find stride candidates in a continuous sensor
signal.
Some are able to directly detect individual biomechanical events.
Other algorithm are only able to detect stride candidates and need to be paired by an explicit event detection
algorithm, as implemented in :py:mod:`gaitmap.event_detection`, to be able to provide information about biomechanical
events.
"""

from gaitmap_mad.stride_segmentation._barth_dtw import BarthDtw
from gaitmap_mad.stride_segmentation._base_dtw import BaseDtw, find_matches_find_peaks, find_matches_min_under_threshold
from gaitmap_mad.stride_segmentation._constrained_barth_dtw import ConstrainedBarthDtw
from gaitmap_mad.stride_segmentation._dtw_templates import (
    BarthOriginalTemplate,
    BaseDtwTemplate,
    DtwTemplate,
    InterpolatedDtwTemplate,
    TrainableTemplateMixin,
)

__all__ = [
    "BarthDtw",
    "ConstrainedBarthDtw",
    "BaseDtw",
    "BaseDtwTemplate",
    "InterpolatedDtwTemplate",
    "DtwTemplate",
    "BarthOriginalTemplate",
    "TrainableTemplateMixin",
    "find_matches_find_peaks",
    "find_matches_min_under_threshold"
]
