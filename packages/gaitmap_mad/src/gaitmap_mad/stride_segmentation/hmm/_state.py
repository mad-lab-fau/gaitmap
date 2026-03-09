"""Serializable HMM state objects and pomegranate conversion helpers."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Union

import numpy as np
import pomegranate as pg
from pomegranate import HiddenMarkovModel as pgHMM
from typing_extensions import Literal

from gaitmap.base import _BaseSerializable
from gaitmap_mad.stride_segmentation.hmm._utils import add_transition


def _get_pomegranate_version() -> str | None:
    try:
        return version("pomegranate")
    except PackageNotFoundError:
        return None


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


def _normalize_mixture_weights(log_weights: np.ndarray) -> np.ndarray:
    shifted = np.exp(log_weights - np.max(log_weights))
    return shifted / np.sum(shifted)


def _distribution_to_state(distribution: pg.Distribution) -> EmissionState:
    if isinstance(distribution, pg.GeneralMixtureModel):
        return GaussianMixtureEmissionState(
            weights=_normalize_mixture_weights(np.asarray(distribution.weights, dtype=float)),
            components=tuple(_distribution_to_state(component) for component in distribution.distributions),  # type: ignore[arg-type]
            frozen=bool(getattr(distribution, "frozen", False)),
        )
    if isinstance(distribution, pg.MultivariateGaussianDistribution):
        mean, covariance = distribution.parameters
        return GaussianEmissionState(
            mean=np.asarray(mean, dtype=float),
            covariance=np.asarray(covariance, dtype=float),
            frozen=bool(getattr(distribution, "frozen", False)),
        )
    raise TypeError(
        f"Unsupported pomegranate emission distribution `{type(distribution).__name__}`. "
        "Only multivariate Gaussian and Gaussian mixture emissions are supported in the serialized HMM state."
    )


def _state_to_distribution(state: EmissionState) -> pg.Distribution:
    if isinstance(state, GaussianEmissionState):
        distribution = pg.MultivariateGaussianDistribution(state.mean.tolist(), state.covariance.tolist())
        distribution.frozen = state.frozen
        return distribution
    if isinstance(state, GaussianMixtureEmissionState):
        weights = np.asarray(state.weights, dtype=float)
        weights = np.clip(weights, np.finfo(float).tiny, None)
        weights = weights / np.sum(weights)
        distribution = pg.GeneralMixtureModel(
            [_state_to_distribution(component) for component in state.components],
            weights=weights.tolist(),
        )
        distribution.frozen = state.frozen
        return distribution
    raise TypeError(f"Unsupported serialized emission state `{type(state).__name__}`.")


def pomegranate_model_to_flat_hmm_state(model: pgHMM) -> FlatHmmState:
    """Convert a pomegranate HMM into a serializable flat state.

    The canonical state only stores emitting states. `pomegranate`'s silent
    start/end nodes are folded into explicit `start_probs`/`end_probs`.
    """
    dense_transition_matrix = model.dense_transition_matrix()
    graph_state = HmmGraphState(
        transition_probs=np.asarray(dense_transition_matrix[:-2, :-2], dtype=float),
        start_probs=np.asarray(dense_transition_matrix[-2, :-2], dtype=float),
        end_probs=np.asarray(dense_transition_matrix[:-2, -1], dtype=float),
    )
    hidden_states = [state for state in model.states if state.distribution is not None]
    return FlatHmmState(
        graph=graph_state,
        emissions=tuple(_distribution_to_state(state.distribution) for state in hidden_states),
        state_names=tuple(state.name for state in hidden_states),
        name=model.name,
    )


def flat_hmm_state_to_pomegranate_model(state: FlatHmmState, *, verbose: bool = False) -> pgHMM:
    """Compile a serializable flat state into a pomegranate HMM."""
    model = pg.HiddenMarkovModel.from_matrix(
        transition_probabilities=np.asarray(state.graph.transition_probs, dtype=float),
        distributions=[_state_to_distribution(distribution) for distribution in state.emissions],
        starts=np.asarray(state.graph.start_probs, dtype=float),
        ends=np.asarray(state.graph.end_probs, dtype=float),
        state_names=list(state.state_names),
        verbose=verbose,
    )
    model.bake()
    if state.name is not None:
        model.name = state.name
    return model


def pomegranate_model_to_hmm_state(
    compiled_model: pgHMM,
    *,
    submodels: tuple[HmmSubModelState, ...] = (),
    cross_module_transitions: tuple[CrossModuleTransition, ...] = (),
    backend_info: BackendInfo | None = None,
) -> HMMState:
    """Convert a compiled pomegranate HMM and optional hierarchy into a serializable HMM state."""
    if backend_info is None:
        backend_info = BackendInfo(backend_id="pomegranate-legacy", backend_version=_get_pomegranate_version())
    elif backend_info.backend_version is None and backend_info.backend_id.startswith("pomegranate"):
        backend_info = BackendInfo(
            backend_id=backend_info.backend_id,
            backend_version=_get_pomegranate_version(),
            state_schema_version=backend_info.state_schema_version,
        )
    return HMMState(
        trained_with=backend_info,
        compiled=pomegranate_model_to_flat_hmm_state(compiled_model),
        submodels=submodels,
        cross_module_transitions=cross_module_transitions,
    )


def hmm_state_to_pomegranate_model(state: HMMState, *, verbose: bool = False) -> pgHMM:
    """Compile a serializable HMM state into a pomegranate HMM."""
    model = flat_hmm_state_to_pomegranate_model(state.compiled, verbose=verbose)
    existing_transitions = {(start.name, end.name) for start, end in model.graph.edges()}
    for transition in state.cross_module_transitions:
        from_state = state.compiled.state_names[_find_state_index(state, transition.from_module, transition.from_state)]
        to_state = state.compiled.state_names[_find_state_index(state, transition.to_module, transition.to_state)]
        if (from_state, to_state) in existing_transitions:
            continue
        add_transition(model, (from_state, to_state), transition.probability)
        existing_transitions.add((from_state, to_state))
    model.bake()
    return model


def _find_state_index(state: HMMState, module_name: str, state_idx: int) -> int:
    offset = 0
    for submodel in state.submodels:
        n_states = len(submodel.model.state_names)
        if submodel.name == module_name:
            if state_idx >= n_states:
                raise ValueError(
                    f"Cross-module transition refers to state {state_idx} of module `{module_name}`, "
                    f"but the module only has {n_states} states."
                )
            return offset + state_idx
        offset += n_states
    raise ValueError(f"No submodel named `{module_name}` exists in the serialized HMM state.")
