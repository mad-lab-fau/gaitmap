"""Temporary module for future features."""
import warnings

warnings.warn(
    "All pipeline features in gaitmap are depcrecated and moved to `tpcp`. Please use tpcp directly. In "
    "most cases this just requires a change of import paths.",
    DeprecationWarning,
)
