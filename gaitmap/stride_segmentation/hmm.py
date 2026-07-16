"""Hidden-Markov based stride segmentation developed by Roth et al."""

from gaitmap.utils._gaitmap_mad import patch_gaitmap_mad_import

_gaitmap_mad_modules = {
    "BaseHmmFeatureTransformer",
    "RothHmmFeatureTransformer",
    "HmmModel",
    "HmmStrideSegmentation",
    "LegacyPomegranateHmmInference",
    "PomegranateHmmInference",
    "PomegranateHmmTrainer",
    "RothSegmentationHmm",
    "ScipyHmmInference",
}

if not (__getattr__ := patch_gaitmap_mad_import(_gaitmap_mad_modules, __name__)):
    del __getattr__
    from gaitmap_mad.stride_segmentation.hmm import (
        BaseHmmFeatureTransformer,
        HmmModel,
        HmmStrideSegmentation,
        LegacyPomegranateHmmInference,
        PomegranateHmmInference,
        PomegranateHmmTrainer,
        RothHmmFeatureTransformer,
        RothSegmentationHmm,
        ScipyHmmInference,
    )


__all__ = [
    "BaseHmmFeatureTransformer",
    "HmmModel",
    "HmmStrideSegmentation",
    "LegacyPomegranateHmmInference",
    "PomegranateHmmInference",
    "PomegranateHmmTrainer",
    "RothHmmFeatureTransformer",
    "RothSegmentationHmm",
    "ScipyHmmInference",
]
