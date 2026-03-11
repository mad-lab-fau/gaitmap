"""Modern-backend-specific training helpers."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from gaitmap.base import _BaseSerializable
from gaitmap_mad.stride_segmentation.hmm._state import (
    BackendInfo,
    FlatHmmState,
    GaussianEmissionState,
    GaussianMixtureEmissionState,
    HmmGraphState,
    HMMState,
)
from gaitmap_mad.stride_segmentation.hmm._utils import (
    cluster_data_by_labels,
    create_state_names,
    create_transition_matrix_fully_connected,
    create_transition_matrix_left_right,
)


def _get_pomegranate_version() -> str | None:
    try:
        return version("pomegranate")
    except PackageNotFoundError:
        return None


class PomegranateModernHistory(_BaseSerializable):
    """Minimal training trace for the modern pomegranate backend."""

    improvements: tuple[float, ...]

    def __init__(self, improvements: tuple[float, ...] = ()) -> None:
        self.improvements = improvements


def fit_gaussian_emission(
    cluster: np.ndarray, n_components: int
) -> GaussianEmissionState | GaussianMixtureEmissionState:
    """Fit one Gaussian or Gaussian-mixture emission to a clustered dataset."""
    if len(cluster) < n_components:
        raise ValueError(
            f"The training labels did only provide {len(cluster)} samples for one state, "
            f"but {n_components} GMM components were requested."
        )
    if cluster.ndim == 1:
        cluster = cluster.reshape(-1, 1)
    mixture = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=0,
        n_init=5,
        reg_covar=np.finfo(float).eps,
    )
    mixture.fit(cluster)
    if n_components == 1:
        return GaussianEmissionState(mean=mixture.means_[0], covariance=mixture.covariances_[0])
    return GaussianMixtureEmissionState(
        weights=mixture.weights_,
        components=tuple(
            GaussianEmissionState(mean=mean, covariance=covariance)
            for mean, covariance in zip(mixture.means_, mixture.covariances_)
        ),
    )


def to_training_arrays(
    data_sequence: list[pd.DataFrame] | list[np.ndarray],
    *,
    data_columns: tuple[str, ...] | None = None,
) -> list[np.ndarray]:
    """Convert feature-space training data to contiguous numpy arrays."""
    arrays = []
    for data in data_sequence:
        if isinstance(data, pd.DataFrame):
            columns = data.columns if data_columns is None else list(data_columns)
            arrays.append(np.ascontiguousarray(data[columns].to_numpy().copy()))
            continue
        arrays.append(np.ascontiguousarray(data.copy()))
    return arrays


def to_modern_input(data_sequence: list[np.ndarray]) -> list[np.ndarray]:
    """Normalize training arrays to the dtype expected by modern pomegranate."""
    return [sequence.astype(float, copy=False) for sequence in data_sequence]


def labels_to_priors(labels_sequence: list[np.ndarray], n_states: int) -> list[np.ndarray]:
    """Convert hard labels into one-hot prior matrices for modern pomegranate."""
    priors = []
    for labels in labels_sequence:
        one_hot = np.zeros((len(labels), n_states), dtype=float)
        one_hot[np.arange(len(labels)), labels.astype(int)] = 1.0
        priors.append(one_hot)
    return priors


def freeze_emission(
    emission: GaussianEmissionState | GaussianMixtureEmissionState,
) -> GaussianEmissionState | GaussianMixtureEmissionState:
    """Clone an emission and mark it as frozen for combined-model training."""
    if isinstance(emission, GaussianEmissionState):
        return GaussianEmissionState(
            mean=emission.mean.copy(),
            covariance=emission.covariance.copy(),
            covariance_type=emission.covariance_type,
            frozen=True,
        )
    return GaussianMixtureEmissionState(
        weights=emission.weights.copy(),
        components=tuple(freeze_emission(component) for component in emission.components),
        frozen=True,
    )


def create_initial_graph_state(
    n_states: int,
    architecture: Literal["left-right-strict", "left-right-loose", "fully-connected"],
    normalize_transition_and_end_probs,
) -> HmmGraphState:
    """Create an initial graph state for modern backend training."""
    if architecture == "left-right-strict":
        transition_matrix, start_probs, end_probs = create_transition_matrix_left_right(n_states, self_transition=False)
    elif architecture == "left-right-loose":
        transition_matrix, _, _ = create_transition_matrix_left_right(n_states, self_transition=True)
        start_probs = np.ones(n_states).astype(float)
        end_probs = np.ones(n_states).astype(float)
    else:
        transition_matrix, start_probs, end_probs = create_transition_matrix_fully_connected(n_states)
    transition_matrix, end_probs = normalize_transition_and_end_probs(transition_matrix, end_probs)
    start_probs = np.asarray(start_probs, dtype=float)
    start_probs /= start_probs.sum()
    return HmmGraphState(transition_probs=transition_matrix, start_probs=start_probs, end_probs=end_probs)


def trainable_state_from_clusters(
    data_sequence: list[np.ndarray],
    labels_sequence: list[np.ndarray],
    *,
    n_states: int,
    n_gmm_components: int,
    architecture: Literal["left-right-strict", "left-right-loose", "fully-connected"],
    name: str,
    normalize_transition_and_end_probs,
) -> FlatHmmState:
    """Create an initial trainable flat state from clustered labeled data."""
    clustered_data = cluster_data_by_labels(data_sequence, labels_sequence)
    if len(clustered_data) < n_states:
        raise ValueError(
            f"The training labels did only provide samples for {len(clustered_data)} states, but {n_states} states "
            "were expected."
        )
    emissions = tuple(fit_gaussian_emission(cluster, n_gmm_components) for cluster in clustered_data[:n_states])
    return FlatHmmState(
        graph=create_initial_graph_state(n_states, architecture, normalize_transition_and_end_probs),
        emissions=emissions,
        state_names=create_state_names(n_states),
        name=name,
    )


def build_tmp_hmm_state(model: FlatHmmState, backend_id: str) -> HMMState:
    """Wrap a flat state in a temporary `HMMState` container."""
    return HMMState(
        trained_with=BackendInfo(backend_id=backend_id, backend_version=_get_pomegranate_version()),
        compiled=model,
    )
