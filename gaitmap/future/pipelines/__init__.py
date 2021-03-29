"""A collection of higher level algorithms to run, optimize and validate algorithms."""
from gaitmap.future.pipelines._optimize import GridSearch, Optimize, BaseOptimize, GridSearchCV
from gaitmap.future.pipelines._validation import cross_validate
from gaitmap.future.pipelines._pipelines import SimplePipeline, OptimizablePipeline
from gaitmap.future.pipelines._scorer import GaitmapScorer

__all__ = [
    "SimplePipeline",
    "OptimizablePipeline",
    "GaitmapScorer",
    "BaseOptimize",
    "GridSearch",
    "GridSearchCV" "Optimize",
    "cross_validate",
]
