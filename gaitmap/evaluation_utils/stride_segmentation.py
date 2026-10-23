"""Compatibility imports for the algorithm-specific evaluation module."""

from gaitmap.stride_segmentation.evaluation import (
    evaluate_segmented_stride_list,
    match_stride_lists,
)

__all__ = ["evaluate_segmented_stride_list", "match_stride_lists"]
