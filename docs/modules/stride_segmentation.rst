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

.. autosummary::
   :toctree: generated/stride_segmentation/hmm
   :template: class.rst

    RothHMM
    SegmentationHMM
    SimpleHMM
    RothHMMFeatureTransformer
    PreTrainedRothSegmentationModel
    HMMFeatureTransformer
