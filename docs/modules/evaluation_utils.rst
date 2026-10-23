gaitmap.evaluation_utils: Helper to evaluate the output of the algorithms
================================================================================

.. automodule:: gaitmap.evaluation_utils
    :no-members:
    :no-inherited-members:

Functions
---------
.. currentmodule:: gaitmap.evaluation_utils

.. autosummary::
   :toctree: generated/evaluation_utils
   :template: function.rst

    evaluate_segmented_stride_list
    match_stride_lists
    recall_score
    precision_score
    f1_score
    precision_recall_f1_score
    evaluate_stride_event_list
    calculate_parameter_errors
    calculate_aggregated_parameter_errors

The preferred imports are now in :mod:`gaitmap.stride_segmentation.evaluation`,
:mod:`gaitmap.event_detection.evaluation`, and :mod:`gaitmap.parameters.evaluation`.
The imports in this module remain supported.
