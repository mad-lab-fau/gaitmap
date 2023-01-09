"""A set of methods to detect static regions/zero-velocity regions (ZUPTS) in a signal."""
from gaitmap.zupt_detection._moving_window_zupt_detector import AredZuptDetector, NormZuptDetector, ShoeZuptDetector

__all__ = ["NormZuptDetector", "ShoeZuptDetector", "AredZuptDetector"]
