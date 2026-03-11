"""Legacy pomegranate adapters for serializable HMM states."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

try:
    import pomegranate as pg
except ImportError:  # pragma: no cover - exercised in environments without pomegranate
    pg = None

from gaitmap_mad.stride_segmentation.hmm._state import (
    BackendInfo,
    CrossModuleTransition,
    EmissionState,
    FlatHmmState,
    GaussianEmissionState,
    GaussianMixtureEmissionState,
    HmmGraphState,
    HMMState,
    HmmSubModelState,
)
from gaitmap_mad.stride_segmentation.hmm.legacy._utils import add_transition


def _get_pomegranate_version() -> str | None:
    try:
        return version("pomegranate")
    except PackageNotFoundError:
        return None


def _require_legacy_pomegranate():
    legacy_hmm = getattr(pg, "HiddenMarkovModel", None)
    if pg is None or legacy_hmm is None:
        raise ImportError("The legacy HMM backend requires pomegranate 0.x with `HiddenMarkovModel` support.")
    return pg


def _normalize_mixture_weights(log_weights: np.ndarray) -> np.ndarray:
    shifted = np.exp(log_weights - np.max(log_weights))
    return shifted / np.sum(shifted)


def _legacy_distribution_to_state(distribution: Any) -> EmissionState:
    pg = _require_legacy_pomegranate()
    if isinstance(distribution, pg.GeneralMixtureModel):
        return GaussianMixtureEmissionState(
            weights=_normalize_mixture_weights(np.asarray(distribution.weights, dtype=float)),
            components=tuple(_legacy_distribution_to_state(component) for component in distribution.distributions),
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


def _state_to_legacy_distribution(state: EmissionState) -> Any:
    pg = _require_legacy_pomegranate()
    if isinstance(state, GaussianEmissionState):
        distribution = pg.MultivariateGaussianDistribution(state.mean.tolist(), state.covariance.tolist())
        distribution.frozen = state.frozen
        return distribution
    if isinstance(state, GaussianMixtureEmissionState):
        weights = np.asarray(state.weights, dtype=float)
        weights = np.clip(weights, np.finfo(float).tiny, None)
        weights = weights / np.sum(weights)
        distribution = pg.GeneralMixtureModel(
            [_state_to_legacy_distribution(component) for component in state.components],
            weights=weights.tolist(),
        )
        distribution.frozen = state.frozen
        return distribution
    raise TypeError(f"Unsupported serialized emission state `{type(state).__name__}`.")


def pomegranate_model_to_flat_hmm_state(model: Any) -> FlatHmmState:
    dense_transition_matrix = model.dense_transition_matrix()
    graph_state = HmmGraphState(
        transition_probs=np.asarray(dense_transition_matrix[:-2, :-2], dtype=float),
        start_probs=np.asarray(dense_transition_matrix[-2, :-2], dtype=float),
        end_probs=np.asarray(dense_transition_matrix[:-2, -1], dtype=float),
    )
    hidden_states = [state for state in model.states if state.distribution is not None]
    return FlatHmmState(
        graph=graph_state,
        emissions=tuple(_legacy_distribution_to_state(state.distribution) for state in hidden_states),
        state_names=tuple(state.name for state in hidden_states),
        name=model.name,
    )


def flat_hmm_state_to_pomegranate_model(state: FlatHmmState, *, verbose: bool = False) -> Any:
    pg = _require_legacy_pomegranate()
    model = pg.HiddenMarkovModel.from_matrix(
        transition_probabilities=np.asarray(state.graph.transition_probs, dtype=float),
        distributions=[_state_to_legacy_distribution(distribution) for distribution in state.emissions],
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
    compiled_model: Any,
    *,
    submodels: tuple[HmmSubModelState, ...] = (),
    cross_module_transitions: tuple[CrossModuleTransition, ...] = (),
    backend_info: BackendInfo | None = None,
) -> HMMState:
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


def hmm_state_to_pomegranate_model(state: HMMState, *, verbose: bool = False) -> Any:
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
