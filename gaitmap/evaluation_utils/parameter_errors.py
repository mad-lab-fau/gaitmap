"""Compatibility imports for the algorithm-specific evaluation module."""

from gaitmap.parameters.evaluation import (
    calculate_aggregated_parameter_errors,
    calculate_parameter_errors,
)

__all__ = ["calculate_aggregated_parameter_errors", "calculate_parameter_errors"]
