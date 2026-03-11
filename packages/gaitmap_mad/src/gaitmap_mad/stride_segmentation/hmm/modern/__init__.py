"""Modern pomegranate backend package."""

from gaitmap_mad.stride_segmentation.hmm.modern._backend import (
    DenseHMM,
    PomegranateModernHmmBackend,
    _get_pomegranate_version,
    torch,
)

if DenseHMM is None:
    raise ImportError(
        "The modern HMM backend requires `pomegranate 1.x` with `DenseHMM` support. "
        f"Installed version: {_get_pomegranate_version() or 'not installed'}."
    )
if torch is None:
    raise ImportError(
        "The modern HMM backend requires `torch` because `pomegranate 1.x` training and inference run on PyTorch."
    )

__all__ = ["PomegranateModernHmmBackend"]
