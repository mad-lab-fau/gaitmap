"""Legacy pomegranate backend package."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

import pomegranate as pg


def _get_pomegranate_version() -> str | None:
    try:
        return version("pomegranate")
    except PackageNotFoundError:
        return None


if getattr(pg, "HiddenMarkovModel", None) is None:
    raise ImportError(
        "The legacy HMM backend requires `pomegranate 0.x` with `HiddenMarkovModel` support. "
        f"Installed version: {_get_pomegranate_version() or 'not installed'}."
    )

_backend = import_module("gaitmap_mad.stride_segmentation.hmm.legacy._backend")
PomegranateLegacyHmmBackend = _backend.PomegranateLegacyHmmBackend
initialize_hmm = _backend.initialize_hmm

__all__ = ["PomegranateLegacyHmmBackend", "initialize_hmm"]
