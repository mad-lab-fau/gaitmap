"""Reusable gait analysis pipelines and their evaluation challenges."""

from gaitmap.pipeline._generic_pipeline import GenericGaitPipeline
from gaitmap.pipeline.base import BaseGaitDataset

__all__ = ["BaseGaitDataset", "GenericGaitPipeline"]
