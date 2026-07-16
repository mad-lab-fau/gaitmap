"""Roth et al. HMM based stride segmentation."""

from gaitmap_mad.stride_segmentation.hmm._hmm_feature_transform import (
    BaseHmmFeatureTransformer,
    RothHmmFeatureTransformer,
)
from gaitmap_mad.stride_segmentation.hmm._hmm_stride_segmentation import HmmStrideSegmentation
from gaitmap_mad.stride_segmentation.hmm._legacy_pomegranate import LegacyPomegranateHmmInference
from gaitmap_mad.stride_segmentation.hmm._model import HmmModel
from gaitmap_mad.stride_segmentation.hmm._pomegranate_backend import (
    PomegranateHmmInference,
    PomegranateHmmTrainer,
)
from gaitmap_mad.stride_segmentation.hmm._roth_model import RothSegmentationHmm
from gaitmap_mad.stride_segmentation.hmm._scipy_inference import ScipyHmmInference

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
