"""Dtw based Stride Segmentation."""

from gaitmap_mad.stride_segmentation.dtw._barth_dtw import BarthDtw
from gaitmap_mad.stride_segmentation.dtw._base_dtw import (
    BaseDtw,
    find_matches_find_peaks,
    find_matches_min_under_threshold,
)
from gaitmap_mad.stride_segmentation.dtw._constrained_barth_dtw import ConstrainedBarthDtw
from gaitmap_mad.stride_segmentation.dtw._dtw_templates.templates import (
    BarthOriginalTemplate,
    BaseDtwTemplate,
    DtwTemplate,
    InterpolatedDtwTemplate,
    TrainableTemplateMixin,
)
from gaitmap_mad.stride_segmentation.dtw._incorrect_but_better_constrained_barth_dtw import (
    IncorrectButBetterConstrainedBarthDtw,
)

__all__ = [
    "BarthDtw",
    "BarthOriginalTemplate",
    "BaseDtw",
    "BaseDtwTemplate",
    "ConstrainedBarthDtw",
    "DtwTemplate",
    "IncorrectButBetterConstrainedBarthDtw",
    "InterpolatedDtwTemplate",
    "TrainableTemplateMixin",
    "find_matches_find_peaks",
    "find_matches_min_under_threshold",
]
