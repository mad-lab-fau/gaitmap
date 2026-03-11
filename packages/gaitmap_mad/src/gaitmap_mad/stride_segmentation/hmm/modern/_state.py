"""Modern pomegranate adapters for serializable HMM states."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
try:
    from pomegranate.distributions import Normal
except (ImportError, AttributeError):  # pragma: no cover - exercised in environments without modern pomegranate
    Normal = None
try:
    from pomegranate.gmm import GeneralMixtureModel
except (ImportError, AttributeError):  # pragma: no cover - exercised in environments without modern pomegranate
    GeneralMixtureModel = None
try:
    from pomegranate.hmm import DenseHMM
except (ImportError, AttributeError):  # pragma: no cover - exercised in environments without modern pomegranate
    DenseHMM = None

from gaitmap_mad.stride_segmentation.hmm._state import (
    BackendInfo,
    EmissionState,
    FlatHmmState,
    GaussianEmissionState,
    GaussianMixtureEmissionState,
    HMMState,
    HmmGraphState,
    HmmSubModelState,
)
from gaitmap_mad.stride_segmentation.hmm._utils import create_state_names


def _get_pomegranate_version() -> str | None:
    try:
        return version("pomegranate")
    except PackageNotFoundError:
        return None


def _require_modern_pomegranate() -> tuple[Any, Any, Any]:
    if DenseHMM is None or Normal is None or GeneralMixtureModel is None:
        raise ImportError("The modern HMM backend requires pomegranate 1.x with `DenseHMM` support.")
    return DenseHMM, Normal, GeneralMixtureModel


def _parameter_to_numpy(parameter: Any) -> np.ndarray:
    if hasattr(parameter, "detach"):
        return parameter.detach().cpu().numpy()
    return np.asarray(parameter, dtype=float)


def _modern_distribution_to_state(distribution: Any) -> EmissionState:
    _, normal, general_mixture_model = _require_modern_pomegranate()
    if isinstance(distribution, general_mixture_model):
        return GaussianMixtureEmissionState(
            weights=np.asarray(_parameter_to_numpy(distribution.priors), dtype=float),
            components=tuple(_modern_distribution_to_state(component) for component in distribution.distributions),
            frozen=bool(_parameter_to_numpy(distribution.frozen)),
        )
    if isinstance(distribution, normal):
        return GaussianEmissionState(
            mean=np.asarray(_parameter_to_numpy(distribution.means), dtype=float),
            covariance=np.asarray(_parameter_to_numpy(distribution.covs), dtype=float),
            frozen=bool(_parameter_to_numpy(distribution.frozen)),
        )
    raise TypeError(
        f"Unsupported modern pomegranate emission distribution `{type(distribution).__name__}`. "
        "Only Normal and GeneralMixtureModel emissions are supported in the serialized HMM state."
    )


def _state_to_modern_distribution(state: EmissionState) -> Any:
    _, normal, general_mixture_model = _require_modern_pomegranate()
    if isinstance(state, GaussianEmissionState):
        return normal(
            means=np.asarray(state.mean, dtype=float),
            covs=np.asarray(state.covariance, dtype=float),
            covariance_type=state.covariance_type,
            frozen=state.frozen,
        )
    if isinstance(state, GaussianMixtureEmissionState):
        weights = np.asarray(state.weights, dtype=float)
        weights = np.clip(weights, np.finfo(float).tiny, None)
        weights = weights / np.sum(weights)
        return general_mixture_model(
            [_state_to_modern_distribution(component) for component in state.components],
            priors=weights,
            frozen=state.frozen,
        )
    raise TypeError(f"Unsupported serialized emission state `{type(state).__name__}`.")


def pomegranate_modern_model_to_flat_hmm_state(
    model: Any,
    *,
    state_names: tuple[str, ...] | None = None,
    name: str | None = None,
) -> FlatHmmState:
    if state_names is None:
        state_names = create_state_names(len(model.distributions))
    return FlatHmmState(
        graph=HmmGraphState(
            transition_probs=np.exp(np.asarray(_parameter_to_numpy(model.edges), dtype=float)),
            start_probs=np.exp(np.asarray(_parameter_to_numpy(model.starts), dtype=float)),
            end_probs=np.exp(np.asarray(_parameter_to_numpy(model.ends), dtype=float)),
        ),
        emissions=tuple(_modern_distribution_to_state(distribution) for distribution in model.distributions),
        state_names=state_names,
        name=name,
    )


def flat_hmm_state_to_pomegranate_modern_model(
    state: FlatHmmState,
    *,
    verbose: bool = False,
    max_iterations: int = 1000,
    stop_threshold: float = 0.1,
) -> Any:
    dense_hmm, _, _ = _require_modern_pomegranate()
    model = dense_hmm(
        distributions=[_state_to_modern_distribution(distribution) for distribution in state.emissions],
        edges=np.asarray(state.graph.transition_probs, dtype=float),
        starts=np.asarray(state.graph.start_probs, dtype=float),
        ends=np.asarray(state.graph.end_probs, dtype=float),
        max_iter=max_iterations,
        tol=stop_threshold,
        verbose=verbose,
    )
    if state.name is not None:
        model.name = state.name
    return model


def pomegranate_modern_model_to_hmm_state(
    compiled_model: Any,
    *,
    submodels: tuple[HmmSubModelState, ...] = (),
    cross_module_transitions: tuple[Any, ...] = (),
    backend_info: BackendInfo | None = None,
    state_names: tuple[str, ...] | None = None,
    name: str | None = None,
) -> HMMState:
    if backend_info is None:
        backend_info = BackendInfo(backend_id="pomegranate-modern", backend_version=_get_pomegranate_version())
    elif backend_info.backend_version is None and backend_info.backend_id.startswith("pomegranate"):
        backend_info = BackendInfo(
            backend_id=backend_info.backend_id,
            backend_version=_get_pomegranate_version(),
            state_schema_version=backend_info.state_schema_version,
        )
    return HMMState(
        trained_with=backend_info,
        compiled=pomegranate_modern_model_to_flat_hmm_state(compiled_model, state_names=state_names, name=name),
        submodels=submodels,
        cross_module_transitions=cross_module_transitions,
    )


def hmm_state_to_pomegranate_modern_model(
    state: HMMState,
    *,
    verbose: bool = False,
    max_iterations: int = 1000,
    stop_threshold: float = 0.1,
) -> Any:
    return flat_hmm_state_to_pomegranate_modern_model(
        state.compiled,
        verbose=verbose,
        max_iterations=max_iterations,
        stop_threshold=stop_threshold,
    )
