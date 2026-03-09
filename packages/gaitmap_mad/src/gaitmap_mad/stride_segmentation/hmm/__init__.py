"""Roth et al. HMM based stride segmentation model."""

import multiprocessing
import warnings

from gaitmap_mad.stride_segmentation.hmm._backend import BaseHmmBackend, PomegranateHmmBackend
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
from gaitmap_mad.stride_segmentation.hmm._simple_model import SimpleHmm
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

if multiprocessing.parent_process() is None:
    warnings.warn(
        "The hmm support in gaitmap is still quite experimental and you might run into some rough edges. "
        "If you encounter any issues, please report them on github. "
        "Also expect the API to change in the future. "
        "Monitor the changelog before upgrading to newer versions when using HMMs.",
        UserWarning,
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
    "RothHmmConfig",
    "RothHmmFeatureTransformer",
    "RothSegmentationHmm",
    "SimpleHmm",
]
