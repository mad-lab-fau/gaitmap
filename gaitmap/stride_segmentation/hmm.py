"""Hidden-Markov based stride segmentation developed by Roth et al..

The default training and inference backend is currently based on pomegranate [1]_.

.. [1] Schreiber, J. (2018). Pomegranate: fast and flexible probabilistic modeling in python.
   Journal of Machine Learning Research, 18(164), 1-6.

"""

from importlib import import_module
from typing import TYPE_CHECKING

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
    "PomegranateLegacyHmmBackend",
    "PomegranateHmmBackend",
    "PomegranateModernHmmBackend",
    "SimpleHmm",
    "RothHmmConfig",
    "RothSegmentationHmm",
    "ScipyHmmInferenceBackend",
    "PreTrainedRothSegmentationModel",
    "BaseSegmentationHmm",
    "get_default_hmm_backend",
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
        PreTrainedRothSegmentationModel,
        RothHmmConfig,
        RothHmmFeatureTransformer,
        RothSegmentationHmm,
        get_default_hmm_backend,
    )

    if TYPE_CHECKING:
        from gaitmap_mad.stride_segmentation.hmm.legacy import (
            PomegranateLegacyHmmBackend,
        )
        from gaitmap_mad.stride_segmentation.hmm.legacy import (
            PomegranateLegacyHmmBackend as PomegranateHmmBackend,
        )
        from gaitmap_mad.stride_segmentation.hmm.legacy import SimpleHmm
        from gaitmap_mad.stride_segmentation.hmm.modern import PomegranateModernHmmBackend
        from gaitmap_mad.stride_segmentation.hmm.scipy import ScipyHmmInferenceBackend

    def __getattr__(name: str):
        if name in {
            "PomegranateLegacyHmmBackend",
            "PomegranateHmmBackend",
            "PomegranateModernHmmBackend",
            "ScipyHmmInferenceBackend",
        }:
            return getattr(import_module("gaitmap_mad.stride_segmentation.hmm"), name)
        if name == "SimpleHmm":
            return import_module("gaitmap_mad.stride_segmentation.hmm.legacy").SimpleHmm
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "PomegranateLegacyHmmBackend",
    "PomegranateModernHmmBackend",
    "PreTrainedRothSegmentationModel",
    "RothHmmConfig",
    "RothHmmFeatureTransformer",
    "RothSegmentationHmm",
    "ScipyHmmInferenceBackend",
    "SimpleHmm",
    "get_default_hmm_backend",
]
