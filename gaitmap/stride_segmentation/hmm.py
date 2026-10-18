"""Hidden-Markov based stride segmentation developed by Roth et al..

HMM training and inference use a local NumPy/Numba implementation, compatible with saved Gaussian models
from the legacy pomegranate backend [1]_.

.. [1] Schreiber, J. (2018). Pomegranate: fast and flexible probabilistic modeling in python.
   Journal of Machine Learning Research, 18(164), 1-6.

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
