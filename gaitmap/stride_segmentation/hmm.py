"""

"""

from gaitmap.utils._gaitmap_mad import patch_gaitmap_mad_import

_gaitmap_mad_modules = {
    "HMMFeatureTransformer",
    "RothHMMFeatureTransformer",
    "RothHMM",
    "SimpleHMM",
    "SimpleSegmentationHMM",
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
        SimpleSegmentationHMM,
    )


__all__ = [
    "RothHMMFeatureTransformer",
    "RothHMM",
    "SimpleHMM",
    "SimpleSegmentationHMM",
    "PreTrainedRothSegmentationModel",
    "HMMFeatureTransformer",
]
