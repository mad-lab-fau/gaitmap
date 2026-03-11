"""Modern pomegranate adapters for serializable HMM states."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from types import MethodType
from typing import Any

import numpy as np
import torch
from pomegranate._utils import _update_parameter
from pomegranate.distributions import Normal
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.hmm import DenseHMM

from gaitmap_mad.stride_segmentation.hmm._state import (
    BackendInfo,
    EmissionState,
    FlatHmmState,
    GaussianEmissionState,
    GaussianMixtureEmissionState,
    HmmGraphState,
    HMMState,
    HmmSubModelState,
)
from gaitmap_mad.stride_segmentation.hmm._utils import create_state_names


def _get_pomegranate_version() -> str | None:
    try:
        return version("pomegranate")
    except PackageNotFoundError:
        return None


def _require_modern_pomegranate() -> tuple[Any, Any, Any]:
    return DenseHMM, Normal, GeneralMixtureModel


def _parameter_to_numpy(parameter: Any) -> np.ndarray:
    if hasattr(parameter, "detach"):
        return parameter.detach().cpu().numpy()
    return np.asarray(parameter, dtype=float)


# Compatibility shim for pomegranate < the upstream fixes from PR #1143:
# https://github.com/jmschrei/pomegranate/pull/1143
_MODERN_MIN_COVARIANCE = 1e-6


def _stabilize_covariance(covariance: Any, covariance_type: str, min_cov: Any) -> Any:
    min_cov = torch.as_tensor(min_cov, dtype=covariance.dtype, device=covariance.device)
    if covariance_type == "full":
        if not torch.isfinite(covariance).all():
            return min_cov * torch.eye(covariance.shape[-1], dtype=covariance.dtype, device=covariance.device)
        stabilized = 0.5 * (covariance + covariance.transpose(-1, -2))
        min_eigenvalue = torch.linalg.eigvalsh(stabilized).min()
        if not torch.isfinite(min_eigenvalue) or min_eigenvalue < min_cov:
            stabilized = stabilized + (min_cov - min_eigenvalue + min_cov) * torch.eye(
                stabilized.shape[-1], dtype=stabilized.dtype, device=stabilized.device
            )
        return stabilized
    stabilized = torch.nan_to_num(covariance, nan=min_cov.item(), posinf=min_cov.item(), neginf=min_cov.item())
    return torch.maximum(stabilized, min_cov)


def _patch_distribution_numerics(distribution: Any) -> None:
    if getattr(distribution, "_gaitmap_min_cov_patch", False):
        return
    if hasattr(distribution, "distributions"):
        for child in distribution.distributions:
            _patch_distribution_numerics(child)
    if not hasattr(distribution, "covs") or not hasattr(distribution, "from_summaries"):
        return

    original_reset_cache = distribution._reset_cache

    def _reset_cache_with_stable_covariance(self) -> Any:
        if getattr(self, "_initialized", False):
            min_cov = _MODERN_MIN_COVARIANCE if self.min_cov is None else self.min_cov
            with torch.no_grad():
                self.covs.copy_(_stabilize_covariance(self.covs, self.covariance_type, min_cov))
        return original_reset_cache()

    def _from_summaries_with_min_cov(self) -> Any:
        # Mirror the upstream PR #1143 fix that applies `min_cov` during the
        # Normal M-step. Current releases store `min_cov` but do not use it.
        if self.frozen is True:
            return None

        means = self._xw_sum / self._w_sum
        min_cov = (
            None
            if self.min_cov is None
            else torch.as_tensor(self.min_cov, dtype=self.covs.dtype, device=self.covs.device)
        )

        if self.covariance_type == "full":
            v = self._xw_sum.unsqueeze(0) * self._xw_sum.unsqueeze(1)
            covs = self._xxw_sum / self._w_sum - v / self._w_sum**2.0
            covs = 0.5 * (covs + covs.transpose(-1, -2))
            if min_cov is not None:
                covs = covs + min_cov * torch.eye(covs.shape[-1], dtype=covs.dtype, device=covs.device)
        elif self.covariance_type in ["diag", "sphere"]:
            covs = self._xxw_sum / self._w_sum - self._xw_sum**2.0 / self._w_sum**2.0
            if self.covariance_type == "sphere":
                covs = covs.mean(dim=-1)
            if min_cov is not None:
                covs = torch.maximum(covs, min_cov)
        else:  # pragma: no cover - mirrors pomegranate's supported covariance types
            raise ValueError(f"Unsupported covariance type `{self.covariance_type}`.")

        if not torch.isfinite(means).all() or not torch.isfinite(covs).all():
            self._reset_cache()
            return None

        _update_parameter(self.means, means, self.inertia)
        _update_parameter(self.covs, covs, self.inertia)
        self._reset_cache()
        return None

    distribution._reset_cache = MethodType(_reset_cache_with_stable_covariance, distribution)
    distribution.from_summaries = MethodType(_from_summaries_with_min_cov, distribution)
    distribution._gaitmap_min_cov_patch = True


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
            min_cov=_MODERN_MIN_COVARIANCE,
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
    # PR #1143 also fixes dtype propagation in the runtime model. Until that is
    # released, cast the constructed DenseHMM explicitly to float64 here.
    model = model.to(torch.float64)
    for distribution in model.distributions:
        _patch_distribution_numerics(distribution)
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
