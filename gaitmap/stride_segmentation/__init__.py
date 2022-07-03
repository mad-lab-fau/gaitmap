"""The :py:mod:`gaitmap.stride_segmentation` contains popular algorithms to detect strides in a sensor signal.

The stride segmentation module includes all algorithms that are able to find stride candidates in a continuous sensor
signal.
Some are able to directly detect individual biomechanical events.
Other algorithm are only able to detect stride candidates and need to be paired by an explicit event detection
algorithm, as implemented in :py:mod:`gaitmap.event_detection`, to be able to provide information about biomechanical
events.
"""
from importlib.util import find_spec

from gaitmap.stride_segmentation._roi_stride_segmentation import RoiStrideSegmentation

_gaitmap_mad_modules = {
    "BarthDtw",
    "ConstrainedBarthDtw",
    "BaseDtw",
    "BaseDtwTemplate",
    "InterpolatedDtwTemplate",
    "DtwTemplate",
    "BarthOriginalTemplate",
    "TrainableTemplateMixin",
}


def patch_gaitmap_mad_import(_gaitmap_mad_modules):
    if bool(find_spec("gaitmap_mad")):
        return None
    # from gaitmap.utils.exceptions import GaitmapMadImportError
    def new_getattr(name: str):
        if name in _gaitmap_mad_modules:
            raise ValueError(name, __name__)
        return globals()[name]

    return new_getattr


if overwrite := patch_gaitmap_mad_import(_gaitmap_mad_modules):
    __getattr__ = overwrite
else:
    from gaitmap_mad.stride_segmentation import (
        BarthDtw,
        BarthOriginalTemplate,
        BaseDtw,
        BaseDtwTemplate,
        ConstrainedBarthDtw,
        DtwTemplate,
        InterpolatedDtwTemplate,
        TrainableTemplateMixin,
    )


__all__ = [
    "BarthDtw",
    "ConstrainedBarthDtw",
    "BaseDtw",
    "BaseDtwTemplate",
    "InterpolatedDtwTemplate",
    "DtwTemplate",
    "BarthOriginalTemplate",
    "TrainableTemplateMixin",
    "RoiStrideSegmentation",
]
