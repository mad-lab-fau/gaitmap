"""A set of methods to detect static regions/zero-velocity regions (ZUPTS) in a signal."""
from gaitmap.zupt_detection._norm_zupt_detector import NormZuptDetector, ShoeZuptDetector

__all__ = ["NormZuptDetector", "ShoeZuptDetector"]
