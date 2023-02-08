"""A set of methods to detect static regions/zero-velocity regions (ZUPTS) in a signal."""
from gaitmap.zupt_detection._base import PerSampleZuptDetectorMixin, RegionZuptDetectorMixin
from gaitmap.zupt_detection._combo_zupt_detector import ComboZuptDetector
from gaitmap.zupt_detection._moving_window_zupt_detector import AredZuptDetector, NormZuptDetector, ShoeZuptDetector
from gaitmap.zupt_detection._stride_event_zupt_detector import StrideEventZuptDetector

__all__ = [
    "NormZuptDetector",
    "ShoeZuptDetector",
    "AredZuptDetector",
    "StrideEventZuptDetector",
    "ComboZuptDetector",
    "PerSampleZuptDetectorMixin",
    "RegionZuptDetectorMixin",
]
