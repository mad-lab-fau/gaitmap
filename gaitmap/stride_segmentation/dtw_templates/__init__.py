"""Helper and preconfigured templates to be used with the provided DTW based methods."""

from gaitmap.stride_segmentation.dtw_templates.templates import (
    create_dtw_template,
    create_interpolated_dtw_template,
    DtwTemplate,
    BarthOriginalTemplate,
)

__all__ = ["create_dtw_template", "create_interpolated_dtw_template", "DtwTemplate", "BarthOriginalTemplate"]
