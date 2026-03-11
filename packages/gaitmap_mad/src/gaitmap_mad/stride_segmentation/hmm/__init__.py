"""Roth et al. HMM based stride segmentation model."""

import multiprocessing
import warnings
from importlib import import_module
from typing import TYPE_CHECKING

from gaitmap_mad.stride_segmentation.hmm._backend import BaseHmmBackend, BaseTrainableHmm, get_default_hmm_backend
from gaitmap_mad.stride_segmentation.hmm._config import CompositeHmmConfig, HmmSubModelConfig, RothHmmConfig
from gaitmap_mad.stride_segmentation.hmm._hmm_feature_transform import (
    BaseHmmFeatureTransformer,
    RothHmmFeatureTransformer,
)
from gaitmap_mad.stride_segmentation.hmm._hmm_stride_segmentation import (
    HmmStrideSegmentation,
    PreTrainedRothSegmentationModel,
)
from gaitmap_mad.stride_segmentation.hmm._segmentation_model import BaseSegmentationHmm, RothSegmentationHmm
from gaitmap_mad.stride_segmentation.hmm._state import (
    BackendInfo,
    CrossModuleTransition,
    FlatHmmState,
    GaussianEmissionState,
    GaussianMixtureEmissionState,
    HmmGraphState,
    HMMState,
    HmmSubModelState,
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

if multiprocessing.parent_process() is None:
    warnings.warn(
        "The hmm support in gaitmap is still quite experimental and you might run into some rough edges. "
        "If you encounter any issues, please report them on github. "
        "Also expect the API to change in the future. "
        "Monitor the changelog before upgrading to newer versions when using HMMs.",
        UserWarning,
    )


def __getattr__(name: str):
    if name in {
        "PomegranateLegacyHmmBackend",
        "PomegranateHmmBackend",
        "PomegranateModernHmmBackend",
        "ScipyHmmInferenceBackend",
    }:
        return getattr(import_module("gaitmap_mad.stride_segmentation.hmm._backend"), name)
    if name == "SimpleHmm":
        return import_module("gaitmap_mad.stride_segmentation.hmm.legacy").SimpleHmm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BackendInfo",
    "BaseHmmBackend",
    "BaseTrainableHmm",
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
