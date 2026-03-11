"""Legacy pomegranate backend package."""

from gaitmap_mad.stride_segmentation.hmm.legacy._backend import (
    PomegranateLegacyHmmBackend,
    _get_pomegranate_version,
    initialize_hmm,
    pg,
)

if getattr(pg, "HiddenMarkovModel", None) is None:
    raise ImportError(
        "The legacy HMM backend requires `pomegranate 0.x` with `HiddenMarkovModel` support. "
        f"Installed version: {_get_pomegranate_version() or 'not installed'}."
    )

__all__ = ["PomegranateLegacyHmmBackend", "initialize_hmm"]
