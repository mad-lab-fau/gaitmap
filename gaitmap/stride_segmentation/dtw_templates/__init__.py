"""Helper and preconfigured templates to be used with the provided DTW based methods."""

from gaitmap.stride_segmentation.dtw_templates.templates import (
    BarthOriginalTemplate,
    DtwTemplate,
    InterpolatedDtwTemplate,
    BaseDtwTemplate,
)

__all__ = ["DtwTemplate", "BarthOriginalTemplate", "InterpolatedDtwTemplate", "BaseDtwTemplate"]
