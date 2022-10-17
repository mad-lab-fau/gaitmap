"""

"""

from gaitmap.utils._gaitmap_mad import patch_gaitmap_mad_import

_gaitmap_mad_modules = {
    "FeatureTransformHMM",
    "RothHMM",
    "SimpleHMM",
    "SimpleSegmentationHMM",
    "PreTrainedSegmentationHMM",
}

if not (__getattr__ := patch_gaitmap_mad_import(_gaitmap_mad_modules, __name__)):
    del __getattr__
    from gaitmap_mad.stride_segmentation.hmm import (
        FeatureTransformHMM,
        RothHMM,
        SimpleHMM,
        SimpleSegmentationHMM,
        PreTrainedSegmentationHMM,
    )


__all__ = ["FeatureTransformHMM", "RothHMM", "SimpleHMM", "SimpleSegmentationHMM", "PreTrainedSegmentationHMM"]
