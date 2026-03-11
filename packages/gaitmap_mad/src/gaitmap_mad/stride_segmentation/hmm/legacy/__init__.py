"""Legacy pomegranate backend package."""

from gaitmap_mad.stride_segmentation.hmm.legacy._backend import (
    PomegranateLegacyHmmBackend,
    SimpleHmm,
    initialize_hmm,
)

__all__ = ["PomegranateLegacyHmmBackend", "SimpleHmm", "initialize_hmm"]
