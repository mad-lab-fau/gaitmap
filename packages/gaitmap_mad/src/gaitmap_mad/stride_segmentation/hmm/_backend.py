"""Lazy facade for HMM backends."""

from __future__ import annotations

from importlib import import_module

from gaitmap_mad.stride_segmentation.hmm._backend_base import BaseHmmBackend, BaseTrainableHmm, get_default_hmm_backend

__all__ = ["BaseHmmBackend", "BaseTrainableHmm", "get_default_hmm_backend"]


def __getattr__(name: str):
    if name == "PomegranateLegacyHmmBackend":
        return import_module("gaitmap_mad.stride_segmentation.hmm.legacy").PomegranateLegacyHmmBackend
    if name == "PomegranateHmmBackend":
        return import_module("gaitmap_mad.stride_segmentation.hmm.legacy").PomegranateLegacyHmmBackend
    if name == "PomegranateModernHmmBackend":
        return import_module("gaitmap_mad.stride_segmentation.hmm.modern").PomegranateModernHmmBackend
    if name == "ScipyHmmInferenceBackend":
        return import_module("gaitmap_mad.stride_segmentation.hmm.scipy").ScipyHmmInferenceBackend
    if name == "SimpleHmm":
        return import_module("gaitmap_mad.stride_segmentation.hmm.legacy").SimpleHmm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
