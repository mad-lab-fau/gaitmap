"""

"""

from gaitmap.utils._gaitmap_mad import patch_gaitmap_mad_import

_gaitmap_mad_modules = {
    "HMMFeatureTransformer",
    "RothHMMFeatureTransformer",
    "RothHMM",
    "SimpleHMM",
    "SegmentationHMM",
    "PreTrainedRothSegmentationModel",
}

if not (__getattr__ := patch_gaitmap_mad_import(_gaitmap_mad_modules, __name__)):
    del __getattr__
    from gaitmap_mad.stride_segmentation.hmm import (
        HMMFeatureTransformer,
        PreTrainedRothSegmentationModel,
        RothHMM,
        RothHMMFeatureTransformer,
        SimpleHMM,
        SegmentationHMM,
    )


__all__ = [
    "RothHMMFeatureTransformer",
    "RothHMM",
    "SimpleHMM",
    "SegmentationHMM",
    "PreTrainedRothSegmentationModel",
    "HMMFeatureTransformer",
]
