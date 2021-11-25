"""Base class for all datasets."""
import warnings

from tpcp import Dataset

warnings.warn(
    "All pipeline features in gaitmap are depcrecated and moved to `tpcp`. Please use tpcp directly. In "
    "most cases this just requires a change of import paths.",
    DeprecationWarning,
)

__all__ = ["Dataset"]
