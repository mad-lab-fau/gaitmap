"""Gaussian emission initialization and weighted re-estimation for local HMM training."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numba import njit
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logsumexp


@dataclass(frozen=True, eq=False, init=False)
class GaussianMixture:
    """One state's Gaussian components, with read-only parameter arrays.

    A single Gaussian uses one component. ``frozen`` prevents all updates;
    ``component_frozen`` fixes individual means and covariances while allowing
    mixture proportions to be re-estimated.
    """

    means: np.ndarray
    covariances: np.ndarray
    weights: np.ndarray
    frozen: bool
    component_frozen: np.ndarray

    def __init__(self, means, covariances, weights, *, frozen=False, component_frozen=None):
        means = np.array(means, dtype=np.float64, copy=True, order="C")
        covariances = np.array(covariances, dtype=np.float64, copy=True, order="C")
        weights = np.array(weights, dtype=np.float64, copy=True, order="C")
        if means.ndim != 2 or not all(means.shape):
            raise ValueError("Gaussian means must have shape (n_components, n_features).")
        n_components, n_features = means.shape
        if covariances.shape != (n_components, n_features, n_features) or weights.shape != (n_components,):
            raise ValueError("Covariances and weights must match the Gaussian components.")
        if not np.isfinite(means).all() or not np.isfinite(covariances).all():
            raise ValueError("Gaussian means and covariances must be finite.")
        if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("Mixture weights must be nonnegative probabilities summing to one.")
        if not np.allclose(covariances, covariances.transpose(0, 2, 1)):
            raise ValueError("Gaussian covariance matrices must be symmetric.")
        try:
            np.linalg.cholesky(covariances)
        except np.linalg.LinAlgError as error:
            raise ValueError("Gaussian covariance matrices must be positive definite.") from error
        component_frozen = (
            np.zeros(n_components, dtype=bool)
            if component_frozen is None
            else np.array(component_frozen, dtype=bool, copy=True)
        )
        if component_frozen.shape != (n_components,):
            raise ValueError("Component freeze flags must match the Gaussian components.")
        for name, array in [
            ("means", means),
            ("covariances", covariances),
            ("weights", weights),
            ("component_frozen", component_frozen),
        ]:
            array.flags.writeable = False
            object.__setattr__(self, name, array)
        object.__setattr__(self, "frozen", bool(frozen))

    @property
    def n_components(self):
        """Return the number of Gaussian components."""
        return self.means.shape[0]

    @property
    def n_features(self):
        """Return the number of observation features."""
        return self.means.shape[1]

    def __getstate__(self):
        return {
            "means": self.means,
            "covariances": self.covariances,
            "weights": self.weights,
            "frozen": self.frozen,
            "component_frozen": self.component_frozen,
        }

    def __setstate__(self, parameters):
        self.__init__(**parameters)

    def __deepcopy__(self, memo):
        # NumPy deepcopy otherwise makes read-only arrays writable again.
        copied = replace(self)
        memo[id(self)] = copied
        return copied


@njit(cache=True, nogil=True)
def _weighted_moments(samples, weights):
    n_samples, dimensions = samples.shape
    anchor = samples[np.argmax(weights)]
    normalized = weights / weights.max()
    total = normalized.sum()
    mean_offset = np.zeros(dimensions)
    for sample in range(n_samples):
        if normalized[sample] == 0:
            continue
        for dimension in range(dimensions):
            mean_offset[dimension] += normalized[sample] * (samples[sample, dimension] - anchor[dimension])
    mean_offset /= total
    covariance = np.zeros((dimensions, dimensions))
    for sample in range(n_samples):
        if normalized[sample] == 0:
            continue
        weight = normalized[sample] / total
        for left in range(dimensions):
            centered_left = (samples[sample, left] - anchor[left]) - mean_offset[left]
            for right in range(left, dimensions):
                centered_right = (samples[sample, right] - anchor[right]) - mean_offset[right]
                covariance[left, right] += weight * centered_left * centered_right
    for left in range(dimensions):
        for right in range(left):
            covariance[left, right] = covariance[right, left]
    return anchor + mean_offset, covariance


def _samples(observations):
    samples = np.asarray(observations, dtype=np.float64)
    return np.ascontiguousarray(samples.reshape(-1, 1) if samples.ndim == 1 else samples)


def _floor_covariance(covariance, min_covar):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floored = covariance
    if eigenvalues[0] < min_covar:
        floored = (eigenvectors * np.maximum(eigenvalues, min_covar)) @ eigenvectors.T
        floored = (floored + floored.T) * 0.5
    try:
        np.linalg.cholesky(floored)
    except np.linalg.LinAlgError:
        # A requested floor far below the largest eigenvalue may disappear when
        # projected back to covariance coordinates. Use its representable scale.
        representable = np.finfo(np.float64).eps * max(abs(eigenvalues)) * len(eigenvalues)
        floor = max(min_covar, representable)
        floored = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
        floored = (floored + floored.T) * 0.5
    return floored


@njit(cache=True, nogil=True)
def _gaussian_scores(samples, means, whitening, normalizers):
    scores = np.empty((len(samples), len(means)))
    dimensions = samples.shape[1]
    for observation in range(len(samples)):
        for component in range(len(means)):
            squared_distance = 0.0
            for row in range(dimensions):
                transformed = 0.0
                for column in range(dimensions):
                    transformed += whitening[component, row, column] * (
                        samples[observation, column] - means[component, column]
                    )
                squared_distance += transformed * transformed
            scores[observation, component] = normalizers[component] - 0.5 * squared_distance
    return scores


def _component_scores(emissions, samples):
    dimensions = emissions.n_features
    whitening = np.empty((emissions.n_components, dimensions, dimensions))
    normalizers = np.empty(emissions.n_components)
    for index, covariance in enumerate(emissions.covariances):
        lower = cholesky(covariance, lower=True)
        whitening[index] = solve_triangular(lower, np.eye(dimensions), lower=True)
        normalizers[index] = -0.5 * dimensions * np.log(2 * np.pi) - np.log(np.diag(lower)).sum()
    return _gaussian_scores(samples, emissions.means, whitening, normalizers)


def update_emissions(emissions, observations, sample_weights, *, min_covar=1e-6):
    """Return weighted emission estimates without changing the input object.

    ``min_covar`` bounds covariance eigenvalues in variance units on every
    estimate. Centered moments and normalized weights keep estimates stable
    for large offsets and tiny positive weights. Only zero-support and frozen
    parameters retain their previous values.
    """
    samples = _samples(observations)
    weights = np.ascontiguousarray(sample_weights, dtype=np.float64)
    if emissions.frozen or not np.any(weights > 0):
        return replace(emissions)
    positive = weights > 0
    samples = np.ascontiguousarray(samples[positive])
    weights = np.ascontiguousarray(weights[positive])
    weights /= weights.max()
    if emissions.n_components == 1:
        responsibilities = weights[:, None]
    else:
        scores = _component_scores(emissions, samples)
        with np.errstate(divide="ignore"):
            scores += np.log(emissions.weights)
        responsibilities = np.exp(scores - logsumexp(scores, axis=1, keepdims=True)) * weights[:, None]
    counts = responsibilities.sum(axis=0)
    means = emissions.means.copy()
    covariances = emissions.covariances.copy()
    for component in range(emissions.n_components):
        if emissions.component_frozen[component] or counts[component] == 0:
            continue
        component_weights = np.ascontiguousarray(responsibilities[:, component])
        means[component], covariance = _weighted_moments(samples, component_weights)
        covariances[component] = _floor_covariance(covariance, min_covar)
    return GaussianMixture(
        means, covariances, counts / counts.sum(), frozen=emissions.frozen, component_frozen=emissions.component_frozen
    )


@njit(cache=True, nogil=True)
def _nearest_centroids(samples, centroids):
    labels = np.empty(len(samples), dtype=np.int64)
    for sample in range(len(samples)):
        best = np.inf
        labels[sample] = 0
        for component in range(len(centroids)):
            distance = 0.0
            for dimension in range(samples.shape[1]):
                distance += (samples[sample, dimension] - centroids[component, dimension]) ** 2
            if distance < best:
                best = distance
                labels[sample] = component
    return labels


def _first_k_labels(samples, n_components):
    centroids = samples[:n_components].copy()
    labels = _nearest_centroids(samples, centroids)
    for component in range(n_components):
        assigned = samples[labels == component]
        if len(assigned):
            centroids[component] = assigned.mean(axis=0)
    return _nearest_centroids(samples, centroids)


def _mixture_likelihood(emissions, samples):
    scores = _component_scores(emissions, samples)
    with np.errstate(divide="ignore"):
        scores += np.log(emissions.weights)
    return logsumexp(scores, axis=1).sum()


def fit_emissions(observations, *, n_components=1, min_covar=1e-6, verbose=False):
    """Fit Gaussian emissions with deterministic first-k clustering and EM.

    One centroid update precedes Gaussian initialization. EM stops after an
    improvement of at most 0.1. Covariance eigenvalues are bounded by
    ``min_covar`` in variance units.
    """
    samples = _samples(observations)
    blank = GaussianMixture(np.zeros((1, samples.shape[1])), np.eye(samples.shape[1])[None, :, :], [1.0])
    if n_components == 1:
        return update_emissions(blank, samples, np.ones(len(samples)), min_covar=min_covar)
    if n_components < 1 or n_components > len(samples):
        raise ValueError("Mixture initialization needs at least one observation per component.")
    labels = _first_k_labels(samples, n_components)
    components = [
        update_emissions(blank, samples[labels == component], np.ones(np.sum(labels == component)), min_covar=min_covar)
        for component in range(n_components)
    ]
    emissions = GaussianMixture(
        np.concatenate([component.means for component in components]),
        np.concatenate([component.covariances for component in components]),
        [float(np.mean(labels == component)) for component in range(n_components)],
    )
    previous = _mixture_likelihood(emissions, samples)
    improvement = np.inf
    while improvement > 0.1:
        emissions = update_emissions(emissions, samples, np.ones(len(samples)), min_covar=min_covar)
        likelihood = _mixture_likelihood(emissions, samples)
        improvement = likelihood - previous
        if verbose:
            print(f"Mixture improvement: {improvement}")
        previous = likelihood
    return emissions
