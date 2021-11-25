"""A collection of higher level algorithms to run, optimize and validate algorithms."""
import warnings

from tpcp import OptimizablePipeline, SimplePipeline
from tpcp.base import BaseOptimize
from tpcp.optimize import GridSearch, GridSearchCV, Optimize
from tpcp.validate import Scorer, cross_validate

warnings.warn(
    "All pipeline features in gaitmap are depcrecated and moved to `tpcp`. Please use tpcp directly. In "
    "most cases this just requires a change of import paths.",
    DeprecationWarning,
)

GaitmapScorer = Scorer


__all__ = [
    "SimplePipeline",
    "OptimizablePipeline",
    "GaitmapScorer",
    "Scorer",
    "BaseOptimize",
    "GridSearch",
    "GridSearchCV",
    "Optimize",
    "cross_validate",
]
