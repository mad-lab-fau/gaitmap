"""Hidden-Markov based stride segmentation developed by Roth et al..

The default training and inference backend is currently based on pomegranate [1]_.

.. [1] Schreiber, J. (2018). Pomegranate: fast and flexible probabilistic modeling in python.
   Journal of Machine Learning Research, 18(164), 1-6.

"""

from gaitmap.utils._gaitmap_mad import patch_gaitmap_mad_import

_gaitmap_mad_modules = {
    "BackendInfo",
    "BaseHmmBackend",
    "CompositeHmmConfig",
    "BaseHmmFeatureTransformer",
    "CrossModuleTransition",
    "FlatHmmState",
    "GaussianEmissionState",
    "GaussianMixtureEmissionState",
    "HMMState",
    "RothHmmFeatureTransformer",
    "HmmStrideSegmentation",
    "HmmGraphState",
    "HmmSubModelConfig",
    "HmmSubModelState",
    "PomegranateHmmBackend",
    "SimpleHmm",
    "RothSegmentationHmm",
    "PreTrainedRothSegmentationModel",
    "BaseSegmentationHmm",
}

if not (__getattr__ := patch_gaitmap_mad_import(_gaitmap_mad_modules, __name__)):
    del __getattr__
    from gaitmap_mad.stride_segmentation.hmm import (
        BackendInfo,
        BaseHmmBackend,
        BaseHmmFeatureTransformer,
        BaseSegmentationHmm,
        CompositeHmmConfig,
        CrossModuleTransition,
        FlatHmmState,
        GaussianEmissionState,
        GaussianMixtureEmissionState,
        HmmGraphState,
        HMMState,
        HmmStrideSegmentation,
        HmmSubModelConfig,
        HmmSubModelState,
        PomegranateHmmBackend,
        PreTrainedRothSegmentationModel,
        RothHmmFeatureTransformer,
        RothSegmentationHmm,
        SimpleHmm,
    )


__all__ = [
    "BackendInfo",
    "BaseHmmBackend",
    "BaseHmmFeatureTransformer",
    "BaseSegmentationHmm",
    "CompositeHmmConfig",
    "CrossModuleTransition",
    "FlatHmmState",
    "GaussianEmissionState",
    "GaussianMixtureEmissionState",
    "HMMState",
    "HmmGraphState",
    "HmmStrideSegmentation",
    "HmmSubModelConfig",
    "HmmSubModelState",
    "PomegranateHmmBackend",
    "PreTrainedRothSegmentationModel",
    "RothHmmFeatureTransformer",
    "RothSegmentationHmm",
    "SimpleHmm",
]
