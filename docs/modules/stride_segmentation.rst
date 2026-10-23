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
    find_matches_find_peaks_with_boundaries
    find_matches_min_under_threshold


HMM Stride Segmentation
***********************
.. currentmodule:: gaitmap.stride_segmentation.hmm

See :ref:`hmm_models` for training and composing HMMs.

.. autosummary::
   :toctree: generated/stride_segmentation/hmm
   :template: class.rst

    HmmStrideSegmentation
    RothSegmentationHmm
    BaseSegmentationHmm
    SimpleHmm
    CompositeHmm
    RothHmmFeatureTransformer
    PreTrainedRothSegmentationModel
    BaseHmmFeatureTransformer
    HiddenMarkovModel
    GaussianMixture
    TrainingHistory

.. autosummary::
   :toctree: generated/stride_segmentation/hmm
   :template: function.rst

    load_legacy_hmm_json


Evaluation
----------

.. currentmodule:: gaitmap.stride_segmentation.evaluation

.. autosummary::
   :toctree: generated/stride_segmentation/evaluation
   :template: function.rst

    evaluate_segmented_stride_list
    match_stride_lists
    precision_recall_f1_score


Challenges
----------

See :doc:`challenges` for installation and usage.

.. currentmodule:: gaitmap.stride_segmentation.challenges

.. autosummary::
   :toctree: generated/stride_segmentation/challenges
   :template: class.rst

    EgaitSegmentationValidation2014Challenge
    EgaitSegmentationValidation2014OriginalLabelChallenge
    SensorPositionComparisonInstepChallenge
