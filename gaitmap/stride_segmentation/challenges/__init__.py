"""Dataset-specific challenges. Install ``gaitmap[challenges]`` to load them."""

from importlib import import_module

_CHALLENGES = {
    "EgaitSegmentationValidation2014Challenge": "egait_segmentation_validation_2014",
    "EgaitSegmentationValidation2014OriginalLabelChallenge": "egait_segmentation_validation_2014_original_label",
    "SensorPositionComparisonInstepChallenge": "sensor_position_comparison_instep",
}

__all__ = list(_CHALLENGES)


def __getattr__(name):
    if name in _CHALLENGES:
        return import_module(f"{__name__}.{_CHALLENGES[name]}").Challenge
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
