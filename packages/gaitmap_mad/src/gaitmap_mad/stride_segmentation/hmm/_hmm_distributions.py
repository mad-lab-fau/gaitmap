"""Gaussian emission initialization and weighted re-estimation for local HMM training."""

import copy

import numpy as np
from numba import njit
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logsumexp


@njit(cache=True, nogil=True)
def _weighted_moments(samples, weights):
    n_samples, dimensions = samples.shape
    total = 0.0
    sums = np.zeros(dimensions)
    products = np.zeros((dimensions, dimensions))
    for sample in range(n_samples):
        weight = weights[sample]
        total += weight
        root_weight = np.sqrt(weight)
        for left in range(dimensions):
            sums[left] += weight * samples[sample, left]
            for right in range(dimensions):
                products[left, right] += (samples[sample, left] * root_weight) * (samples[sample, right] * root_weight)
    return total, sums, products


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


def _update_gaussian(distribution, samples, weights):
    if distribution.get("frozen", False):
        return distribution
    total, sums, products = _weighted_moments(samples, weights)
    dimensions = samples.shape[1]
    if dimensions * total < 1e-7:
        return distribution
    mean = sums / total
    covariance = (products - np.outer(sums, sums) / total) / total
    try:
        cholesky(covariance, lower=True)
    except np.linalg.LinAlgError:
        # Match the legacy MLE repair rather than regularizing every covariance.
        shift = 1e-5 if covariance.sum() == 0 else 1e-8 - np.linalg.eigvalsh(covariance).min()
        covariance += np.eye(dimensions) * shift
    result = copy.deepcopy(distribution)
    result["parameters"] = [mean.tolist(), covariance.tolist()]
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


def update_distribution(serialized_distribution, observations, sample_weights):
    """Return one weighted MLE update, using soft responsibilities inside mixtures.

    Zero-support and frozen distributions retain their parameters. Gaussian
    components with less than ``1e-7`` total dimension weight also retain their
    parameters, matching the legacy backend's sufficient-statistics threshold.
    """
    result = copy.deepcopy(serialized_distribution)
    samples = _samples(observations)
    weights = np.ascontiguousarray(sample_weights, dtype=np.float64)
    if result.get("frozen", False) or weights.sum() == 0:
        return result
    positive = weights > 0
    samples = np.ascontiguousarray(samples[positive])
    weights = np.ascontiguousarray(weights[positive])
    if result["class"] != "GeneralMixtureModel":
        return _update_gaussian(result, samples, weights)
    scores = _component_scores(result["distributions"], samples)
    with np.errstate(divide="ignore"):
        scores += np.log(result["weights"])
    responsibilities = np.exp(scores - logsumexp(scores, axis=1, keepdims=True)) * weights[:, None]
    counts = responsibilities.sum(axis=0)
    result["weights"] = (counts / counts.sum()).tolist()
    result["distributions"] = [
        _update_gaussian(component, samples, np.ascontiguousarray(responsibilities[:, index]))
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


def fit_distribution(observations, n_components=1, verbose=False, n_init=5, n_jobs=1):
    """Initialize Gaussian emissions with deterministic first-k clustering and EM.

    One centroid update precedes Gaussian initialization, as in legacy
    ``max_kmeans_iterations=1``. EM stops after an improvement of at most 0.1.
    First-k restarts produce identical results, so ``n_init`` needs no repeated
    work. ``n_jobs`` is retained for callers; numeric kernels execute locally.
    """
    del n_init, n_jobs
    samples = _samples(observations)
    blank = _gaussian(np.zeros(samples.shape[1]), np.eye(samples.shape[1]))
    if n_components == 1:
        return update_distribution(blank, samples, np.ones(len(samples)))
    if n_components < 1 or n_components > len(samples):
        raise ValueError("Mixture initialization needs at least one observation per component.")
    labels = _first_k_labels(samples, n_components)
    distribution = {
        "class": "GeneralMixtureModel",
        "distributions": [
            update_distribution(blank, samples[labels == component], np.ones(np.sum(labels == component)))
            for component in range(n_components)
        ],
        "weights": [float(np.mean(labels == component)) for component in range(n_components)],
    }
    previous = _mixture_likelihood(distribution, samples)
    improvement = np.inf
    while improvement > 0.1:
        distribution = update_distribution(distribution, samples, np.ones(len(samples)))
        likelihood = _mixture_likelihood(distribution, samples)
        improvement = likelihood - previous
        if verbose:
            print(f"Mixture improvement: {improvement}")
        previous = likelihood
    return distribution
