"""A collection of higher level algorithms to run, optimize and validate algorithms."""
from gaitmap.future.pipelines._optimize import BaseOptimize, GridSearch, GridSearchCV, Optimize
from gaitmap.future.pipelines._pipelines import OptimizablePipeline, SimplePipeline
from gaitmap.future.pipelines._scorer import GaitmapScorer
from gaitmap.future.pipelines._validation import cross_validate

__all__ = [
    "SimplePipeline",
    "OptimizablePipeline",
    "GaitmapScorer",
    "BaseOptimize",
    "GridSearch",
    "GridSearchCV",
    "Optimize",
    "cross_validate",
]
