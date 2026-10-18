gaitmap.stride_segmentation: Algorithms to find stride candidates
=================================================================

.. automodule:: gaitmap.stride_segmentation
    :no-members:
    :no-inherited-members:



Dtw Stride Segmentation
***********************
.. currentmodule:: gaitmap.stride_segmentation

Classes
-------

.. autosummary::
   :toctree: generated/stride_segmentation
   :template: class.rst

    BaseDtw
    BarthDtw
    ConstrainedBarthDtw
    RoiStrideSegmentation
    BaseDtwTemplate
    DtwTemplate
    BarthOriginalTemplate
    InterpolatedDtwTemplate

Functions
---------

.. autosummary::
   :toctree: generated/stride_segmentation
   :template: function.rst

    find_matches_find_peaks
    find_matches_min_under_threshold


HMM Stride Segmentation
***********************
.. currentmodule:: gaitmap.stride_segmentation.hmm

Training uses a local NumPy/Numba backend. Gaussian covariance estimates use
centered differences and weights normalized to remove dependence on their absolute
scale. Each estimate floors its eigenvalues at ``SimpleHmm.min_covar``, which
defaults to ``1e-6`` in squared feature units. The floor prevents mixture components
from collapsing around a few observations. Set it on the stride and transition
submodels separately when their feature scales require different regularization.
Loading an existing model and training with frozen emissions preserve its original
covariance matrices.

.. autosummary::
   :toctree: generated/stride_segmentation/hmm
   :template: class.rst

    HmmStrideSegmentation
    RothSegmentationHmm
    BaseSegmentationHmm
    SimpleHmm
    RothHmmFeatureTransformer
    PreTrainedRothSegmentationModel
    BaseHmmFeatureTransformer
