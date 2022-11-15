"""Roth et al. HMM based stride segmentation model."""
import warnings

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

warnings.warn(
    "The hmm support in gaitmap is still quite experimental and you might run into some rough edges."
    "If you encounter any issues, please report them on github. "
    "Also expect the API to change in the future."
    "Monitor the changelog before upgrading to newer versions when using HMMs.",
    UserWarning,
)

__all__ = [
    "RothHmmFeatureTransformer",
    "HmmStrideSegmentation",
    "SimpleHmm",
    "BaseSegmentationHmm",
    "RothSegmentationHmm",
    "PreTrainedRothSegmentationModel",
    "BaseHmmFeatureTransformer",
]
