"""Hidden-Markov based stride segmentation developed by Roth et al..

Train models with Gaussian or Gaussian-mixture emissions, or load an existing gaitmap HMM model.

"""

from gaitmap.utils._gaitmap_mad import patch_gaitmap_mad_import

_gaitmap_mad_modules = {
    "GaussianMixture",
    "HiddenMarkovModel",
    "TrainingHistory",
    "BaseHmmFeatureTransformer",
    "RothHmmFeatureTransformer",
    "HmmStrideSegmentation",
    "SimpleHmm",
    "RothSegmentationHmm",
    "PreTrainedRothSegmentationModel",
    "BaseSegmentationHmm",
}

if not (__getattr__ := patch_gaitmap_mad_import(_gaitmap_mad_modules, __name__)):
    del __getattr__
    from gaitmap_mad.stride_segmentation.hmm import (
        BaseHmmFeatureTransformer,
        BaseSegmentationHmm,
        GaussianMixture,
        HiddenMarkovModel,
        HmmStrideSegmentation,
        PreTrainedRothSegmentationModel,
        RothHmmFeatureTransformer,
        RothSegmentationHmm,
        SimpleHmm,
        TrainingHistory,
    )


__all__ = [
    "BaseHmmFeatureTransformer",
    "BaseSegmentationHmm",
    "GaussianMixture",
    "HiddenMarkovModel",
    "HmmStrideSegmentation",
    "PreTrainedRothSegmentationModel",
    "RothHmmFeatureTransformer",
    "RothSegmentationHmm",
    "SimpleHmm",
    "TrainingHistory",
]
