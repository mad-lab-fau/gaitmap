"""Dataset-specific challenges. Install ``gaitmap[challenges]`` to load them."""

from importlib import import_module

_CHALLENGES = {
    "EgaitAdidas2014Challenge": "egait_adidas_2014",
    "EgaitParameterValidation2013Challenge": "egait_parameter_validation_2013",
    "SensorPositionComparisonInstepChallenge": "sensor_position_comparison_instep",
}

__all__ = list(_CHALLENGES)


def __getattr__(name):
    if name in _CHALLENGES:
        return import_module(f"{__name__}.{_CHALLENGES[name]}").Challenge
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
