"""Helper and preconfigured templates to be used with the provided DTW based methods."""

from gaitmap_mad.stride_segmentation._dtw_templates.templates import (
    BarthOriginalTemplate,
    BaseDtwTemplate,
    DtwTemplate,
    InterpolatedDtwTemplate,
    TrainableTemplateMixin,
)

__all__ = [
    "DtwTemplate",
    "BarthOriginalTemplate",
    "InterpolatedDtwTemplate",
    "BaseDtwTemplate",
    "TrainableTemplateMixin",
]
