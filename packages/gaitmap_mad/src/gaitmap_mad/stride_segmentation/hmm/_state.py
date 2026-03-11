"""Canonical, backend-neutral HMM state objects."""

from __future__ import annotations

from typing import Union

import numpy as np
from typing_extensions import Literal

from gaitmap.base import _BaseSerializable


class BackendInfo(_BaseSerializable):
    """Provenance information for a serialized HMM state."""

    backend_id: str
    backend_version: str | None
    state_schema_version: int

    def __init__(
        self,
        backend_id: str,
        *,
        backend_version: str | None = None,
        state_schema_version: int = 1,
    ) -> None:
        self.backend_id = backend_id
        self.backend_version = backend_version
        self.state_schema_version = state_schema_version


class HmmGraphState(_BaseSerializable):
    """Dense graph representation of one flat HMM."""

    transition_probs: np.ndarray
    start_probs: np.ndarray
    end_probs: np.ndarray

    def __init__(self, transition_probs: np.ndarray, start_probs: np.ndarray, end_probs: np.ndarray) -> None:
        self.transition_probs = transition_probs
        self.start_probs = start_probs
        self.end_probs = end_probs


class GaussianEmissionState(_BaseSerializable):
    """Serializable multivariate Gaussian emission."""

    kind: Literal["gaussian"]
    mean: np.ndarray
    covariance: np.ndarray
    covariance_type: Literal["full"]
    frozen: bool

    def __init__(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        *,
        covariance_type: Literal["full"] = "full",
        frozen: bool = False,
    ) -> None:
        self.kind = "gaussian"
        self.mean = mean
        self.covariance = covariance
        self.covariance_type = covariance_type
        self.frozen = frozen


class GaussianMixtureEmissionState(_BaseSerializable):
    """Serializable Gaussian-mixture emission."""

    kind: Literal["gaussian_mixture"]
    weights: np.ndarray
    components: tuple[GaussianEmissionState, ...]
    frozen: bool

    def __init__(
        self,
        weights: np.ndarray,
        components: tuple[GaussianEmissionState, ...],
        *,
        frozen: bool = False,
    ) -> None:
        self.kind = "gaussian_mixture"
        self.weights = weights
        self.components = components
        self.frozen = frozen


EmissionState = Union[GaussianEmissionState, GaussianMixtureEmissionState]


class FlatHmmState(_BaseSerializable):
    """Serializable flat HMM representation used for compiled and submodule states."""

    name: str | None
    graph: HmmGraphState
    emissions: tuple[EmissionState, ...]
    state_names: tuple[str, ...]

    def __init__(
        self,
        graph: HmmGraphState,
        emissions: tuple[EmissionState, ...],
        *,
        state_names: tuple[str, ...],
        name: str | None = None,
    ) -> None:
        self.name = name
        self.graph = graph
        self.emissions = emissions
        self.state_names = state_names


class HmmSubModelState(_BaseSerializable):
    """Serializable named submodule state."""

    name: str
    role: str
    model: FlatHmmState

    def __init__(self, name: str, role: str, model: FlatHmmState) -> None:
        self.name = name
        self.role = role
        self.model = model


class CrossModuleTransition(_BaseSerializable):
    """Serializable transition between two named submodules."""

    from_module: str
    from_state: int
    to_module: str
    to_state: int
    probability: float

    def __init__(self, from_module: str, from_state: int, to_module: str, to_state: int, probability: float) -> None:
        self.from_module = from_module
        self.from_state = from_state
        self.to_module = to_module
        self.to_state = to_state
        self.probability = probability


class HMMState(_BaseSerializable):
    """Serializable, backend-neutral trained HMM state."""

    trained_with: BackendInfo
    compiled: FlatHmmState
    submodels: tuple[HmmSubModelState, ...]
    cross_module_transitions: tuple[CrossModuleTransition, ...]

    def __init__(
        self,
        trained_with: BackendInfo,
        compiled: FlatHmmState,
        *,
        submodels: tuple[HmmSubModelState, ...] = (),
        cross_module_transitions: tuple[CrossModuleTransition, ...] = (),
    ) -> None:
        self.trained_with = trained_with
        self.compiled = compiled
        self.submodels = submodels
        self.cross_module_transitions = cross_module_transitions

    def __getstate__(self) -> str:
        """Use the JSON serialization as stable pickle state for hashing and cloning."""
        return self.to_json()

    def __setstate__(self, state: str) -> None:
        restored = type(self).from_json(state)
        self.__dict__.update(restored.__dict__)
