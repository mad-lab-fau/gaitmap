"""A collection of higher level algorithms to run, optimize and validate algorithms."""
from gaitmap.future.pipelines._optimize import GridSearch, Optimize, BaseOptimize
from gaitmap.future.pipelines._pipelines import SimplePipeline, OptimizablePipeline
from gaitmap.future.pipelines._scorer import GaitmapScorer

__all__ = ["SimplePipeline", "OptimizablePipeline", "GaitmapScorer", "BaseOptimize", "GridSearch", "Optimize"]
