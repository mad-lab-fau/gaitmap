"""Structural dataset interface consumed by reusable gait pipelines."""

from typing import Protocol

from gaitmap.utils.datatype_helper import SensorData


class BaseGaitDataset(Protocol):
    """One recording with aligned sensor-frame IMU data and a sampling rate.

    This is a structural protocol. Dataset loaders satisfy it by exposing the
    properties below, without inheriting from gaitmap or depending on it.
    A single-recording subset of a ``tpcp.Dataset`` can satisfy this interface,
    as can an ordinary Python object. Dataset-specific reference labels are
    outside this protocol and belong to the corresponding challenge.
    """

    @property
    def data(self) -> SensorData:
        """Return IMU data aligned to gaitmap's foot sensor frame."""
        ...

    @property
    def sampling_rate_hz(self) -> float:
        """Return the sampling rate of the IMU data in hertz."""
        ...


__all__ = ["BaseGaitDataset"]
