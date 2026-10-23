"""Compatibility access to evaluation helpers, now documented beside each algorithm type."""

from gaitmap.event_detection.evaluation import evaluate_stride_event_list
from gaitmap.parameters.evaluation import calculate_aggregated_parameter_errors, calculate_parameter_errors
from gaitmap.stride_segmentation.evaluation import evaluate_segmented_stride_list, match_stride_lists
from gaitmap.utils._evaluation_scores import f1_score, precision_recall_f1_score, precision_score, recall_score

__all__ = [
    "calculate_aggregated_parameter_errors",
    "calculate_parameter_errors",
    "evaluate_segmented_stride_list",
    "evaluate_stride_event_list",
    "f1_score",
    "match_stride_lists",
    "precision_recall_f1_score",
    "precision_score",
    "recall_score",
]
