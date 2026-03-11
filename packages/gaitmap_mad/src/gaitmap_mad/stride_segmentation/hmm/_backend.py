"""Base backend abstractions and backend selection."""

from gaitmap_mad.stride_segmentation.hmm._backend_base import BaseHmmBackend, BaseTrainableHmm, get_default_hmm_backend

__all__ = ["BaseHmmBackend", "BaseTrainableHmm", "get_default_hmm_backend"]
