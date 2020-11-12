"""The :py:mod:`gaitmap.stride_segmentation` contains popular algorithms to detect strides in a sensor signal.

The stride segmentation module includes all algorithms that are able to find stride candidates in a continuous sensor
signal.
Some are able to directly detect individual biomechanical events.
Other algorithm are only able to detect stride candidates and need to be paired by an explicit event detection
algorithm, as implemented in :py:mod:`gaitmap.event_detection`, to be able to provide information about biomechanical
events.
"""

from gaitmap.stride_segmentation._roi_stride_segmentation import RoiStrideSegmentation
from gaitmap.utils._gaitmap_mad import patch_gaitmap_mad_import

_gaitmap_mad_modules = {
    "BarthDtw",
    "ConstrainedBarthDtw",
    "BaseDtw",
    "BaseDtwTemplate",
    "InterpolatedDtwTemplate",
    "DtwTemplate",
    "BarthOriginalTemplate",
    "TrainableTemplateMixin",
    "find_matches_find_peaks",
    "find_matches_min_under_threshold",
}

if not (__getattr__ := patch_gaitmap_mad_import(_gaitmap_mad_modules, __name__)):
    del __getattr__
    from gaitmap_mad.stride_segmentation import (
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

from gaitmap.stride_segmentation.hmm_models.models import (
    HiddenMarkovModel,
    HiddenMarkovModelStairs,
)
from gaitmap.stride_segmentation.roth_hmm import RothHMM


__all__ = [
    "BarthDtw",
    "ConstrainedBarthDtw",
    "BaseDtw",
    "BaseDtwTemplate",
    "InterpolatedDtwTemplate",
    "DtwTemplate",
    "BarthOriginalTemplate",
    "TrainableTemplateMixin",
    "RoiStrideSegmentation",
    "find_matches_find_peaks",
    "find_matches_min_under_threshold",
    "HiddenMarkovModel",
    "HiddenMarkovModelStairs",
    "RothHMM",
]
