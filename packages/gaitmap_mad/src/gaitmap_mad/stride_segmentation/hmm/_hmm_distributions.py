"""Gaussian emission initialization and weighted re-estimation for local HMM training."""

import copy

import numpy as np
from numba import njit
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logsumexp


@njit(cache=True, nogil=True)
def _weighted_moments(samples, weights):
    n_samples, dimensions = samples.shape
    normalized = weights / weights.max()
    total = normalized.sum()
    mean_offset = np.zeros(dimensions)
    for sample in range(n_samples):
        for dimension in range(dimensions):
            mean_offset[dimension] += normalized[sample] * (samples[sample, dimension] - samples[0, dimension])
    mean_offset /= total
    covariance = np.zeros((dimensions, dimensions))
    for sample in range(n_samples):
        weight = normalized[sample] / total
        for left in range(dimensions):
            centered_left = (samples[sample, left] - samples[0, left]) - mean_offset[left]
            for right in range(left, dimensions):
                centered_right = (samples[sample, right] - samples[0, right]) - mean_offset[right]
                covariance[left, right] += weight * centered_left * centered_right
    for left in range(dimensions):
        for right in range(left):
            covariance[left, right] = covariance[right, left]
    return samples[0] + mean_offset, covariance


def _samples(observations):
    samples = np.asarray(observations, dtype=np.float64)
    return np.ascontiguousarray(samples.reshape(-1, 1) if samples.ndim == 1 else samples)


def _gaussian(mean, covariance):
    return {
        "class": "Distribution",
        "name": "MultivariateGaussianDistribution",
        "parameters": [mean.tolist(), covariance.tolist()],
        "frozen": False,
    }


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


def _update_gaussian(distribution, samples, weights, min_covar):
    if distribution.get("frozen", False) or not np.any(weights > 0):
        return distribution
    mean, covariance = _weighted_moments(samples, weights)
    result = copy.deepcopy(distribution)
    result["parameters"] = [mean.tolist(), _floor_covariance(covariance, min_covar).tolist()]
    return result


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


def _component_scores(distributions, samples):
    dimensions = samples.shape[1]
    means = np.empty((len(distributions), dimensions))
    whitening = np.empty((len(distributions), dimensions, dimensions))
    normalizers = np.empty(len(distributions))
    for index, distribution in enumerate(distributions):
        mean, covariance = distribution["parameters"]
        lower = cholesky(np.asarray(covariance), lower=True)
        means[index] = mean
        whitening[index] = solve_triangular(lower, np.eye(dimensions), lower=True)
        normalizers[index] = -0.5 * dimensions * np.log(2 * np.pi) - np.log(np.diag(lower)).sum()
    # Only the small D-by-D factorization uses SciPy. Applying it to the full
    # observation matrix avoids threaded BLAS setup inside each EM iteration.
    return _gaussian_scores(samples, means, whitening, normalizers)


def update_distribution(serialized_distribution, observations, sample_weights, min_covar=1e-6):
    """Return one weighted MLE update, using soft responsibilities inside mixtures.

    ``min_covar`` bounds covariance eigenvalues in variance units on every
    estimate. Centered moments and normalized weights keep estimates stable
    for large offsets and tiny positive weights. Only zero-support and frozen
    distributions retain their previous parameters.
    """
    result = copy.deepcopy(serialized_distribution)
    samples = _samples(observations)
    weights = np.ascontiguousarray(sample_weights, dtype=np.float64)
    if result.get("frozen", False) or not np.any(weights > 0):
        return result
    positive = weights > 0
    samples = np.ascontiguousarray(samples[positive])
    weights = np.ascontiguousarray(weights[positive])
    weights /= weights.max()
    if result["class"] != "GeneralMixtureModel":
        return _update_gaussian(result, samples, weights, min_covar)
    scores = _component_scores(result["distributions"], samples)
    with np.errstate(divide="ignore"):
        scores += np.log(result["weights"])
    responsibilities = np.exp(scores - logsumexp(scores, axis=1, keepdims=True)) * weights[:, None]
    counts = responsibilities.sum(axis=0)
    result["weights"] = (counts / counts.sum()).tolist()
    result["distributions"] = [
        _update_gaussian(component, samples, np.ascontiguousarray(responsibilities[:, index]), min_covar)
        for index, component in enumerate(result["distributions"])
    ]
    return result


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
            sums = assigned.sum(axis=0)
            # Legacy Kmeans preserves coordinates whose accumulated sum is zero.
            nonzero = sums != 0
            centroids[component, nonzero] = sums[nonzero] / len(assigned)
    return _nearest_centroids(samples, centroids)


def _mixture_likelihood(distribution, samples):
    scores = _component_scores(distribution["distributions"], samples)
    with np.errstate(divide="ignore"):
        scores += np.log(distribution["weights"])
    return logsumexp(scores, axis=1).sum()


def fit_distribution(observations, n_components=1, verbose=False, n_init=5, n_jobs=1, min_covar=1e-6):
    """Initialize Gaussian emissions with deterministic first-k clustering and EM.

    One centroid update precedes Gaussian initialization, as in legacy
    ``max_kmeans_iterations=1``. EM stops after an improvement of at most 0.1.
    Covariance eigenvalues are bounded by ``min_covar`` in variance units.
    First-k restarts produce identical results, so ``n_init`` needs no repeated
    work. ``n_jobs`` is retained for callers; numeric kernels execute locally.
    """
    del n_init, n_jobs
    samples = _samples(observations)
    blank = _gaussian(np.zeros(samples.shape[1]), np.eye(samples.shape[1]))
    if n_components == 1:
        return update_distribution(blank, samples, np.ones(len(samples)), min_covar=min_covar)
    if n_components < 1 or n_components > len(samples):
        raise ValueError("Mixture initialization needs at least one observation per component.")
    labels = _first_k_labels(samples, n_components)
    distribution = {
        "class": "GeneralMixtureModel",
        "distributions": [
            update_distribution(
                blank, samples[labels == component], np.ones(np.sum(labels == component)), min_covar=min_covar
            )
            for component in range(n_components)
        ],
        "weights": [float(np.mean(labels == component)) for component in range(n_components)],
    }
    previous = _mixture_likelihood(distribution, samples)
    improvement = np.inf
    while improvement > 0.1:
        distribution = update_distribution(distribution, samples, np.ones(len(samples)), min_covar=min_covar)
        likelihood = _mixture_likelihood(distribution, samples)
        improvement = likelihood - previous
        if verbose:
            print(f"Mixture improvement: {improvement}")
        previous = likelihood
    return distribution
