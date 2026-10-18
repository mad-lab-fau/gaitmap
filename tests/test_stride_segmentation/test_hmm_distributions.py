"""Gaussian and mixture fitting agree with hand moments, SciPy densities, and numerical invariants."""

import copy
import pickle

import numpy as np
import pytest
from gaitmap_mad.stride_segmentation.hmm._hmm_distributions import fit_emissions, update_emissions
from numpy.testing import assert_allclose
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from gaitmap.stride_segmentation.hmm import GaussianMixture


def assert_emissions_equal(actual, expected):
    """Compare owned numerical parameters without going through serialization."""
    for name in ["means", "covariances", "weights", "component_frozen"]:
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert actual.frozen == expected.frozen


def test_weighted_gaussian_update_matches_hand_computed_full_covariance():
    """Weighted MLE keeps the off-diagonal covariance and leaves the input unchanged."""
    initial = GaussianMixture([[9.0, 9.0]], [np.eye(2)], [1.0])
    observations = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0], [2.0, 3.0]])
    original = copy.deepcopy(initial)

    result = update_emissions(initial, observations, [1.0, 2.0, 3.0, 4.0])

    assert_allclose(result.means, [[1.2, 2.1]], rtol=1e-13)
    assert_allclose(result.covariances, [[[0.96, -0.12], [-0.12, 1.89]]], rtol=1e-13)
    assert_emissions_equal(initial, original)


@pytest.mark.parametrize("frozen_component", [False, True])
def test_mixture_update_matches_scipy_responsibilities_and_weighted_moments(frozen_component):
    """State weights multiply soft component responsibilities, including frozen components."""
    initial = GaussianMixture(
        [[-1.0, 0.0], [1.0, 0.5]],
        [[[1.0, 0.2], [0.2, 0.8]], [[0.7, -0.1], [-0.1, 1.2]]],
        [0.4, 0.6],
        component_frozen=[frozen_component, False],
    )
    observations = np.array([[-2.0, -1.0], [-0.5, 0.7], [0.5, -0.4], [1.5, 1.3], [2.0, 0.2]])
    state_weights = np.array([0.2, 1.0, 0.7, 0.4, 0.8])
    density = np.column_stack(
        [
            multivariate_normal.pdf(observations, mean, covariance) * weight
            for mean, covariance, weight in zip(initial.means, initial.covariances, initial.weights, strict=True)
        ]
    )
    responsibilities = density / density.sum(axis=1, keepdims=True) * state_weights[:, None]
    counts = responsibilities.sum(axis=0)

    result = update_emissions(initial, observations, state_weights)

    assert_allclose(result.weights, counts / counts.sum(), rtol=1e-13)
    for component in range(2):
        if frozen_component and component == 0:
            np.testing.assert_array_equal(result.means[component], initial.means[component])
            np.testing.assert_array_equal(result.covariances[component], initial.covariances[component])
        else:
            expected_mean = np.average(observations, weights=responsibilities[:, component], axis=0)
            centered = observations - expected_mean
            expected_covariance = (centered.T * responsibilities[:, component]) @ centered / counts[component]
            assert_allclose(result.means[component], expected_mean, rtol=1e-13)
            assert_allclose(result.covariances[component], expected_covariance, rtol=1e-13)


def test_zero_probability_component_keeps_its_parameters():
    """A component with no responsibility retains its moments and zero mixture weight."""
    initial = GaussianMixture([[-1.0], [10.0]], [[[1.0]], [[2.0]]], [1.0, 0.0])

    result = update_emissions(initial, [-2.0, -1.0, 0.0], np.ones(3))

    assert_allclose(result.means[:, 0], [-1.0, 10.0])
    assert_allclose(result.covariances[:, 0, 0], [2 / 3, 2.0])
    np.testing.assert_array_equal(result.weights, [1.0, 0.0])


@pytest.mark.parametrize("n_components", [1, 2])
@pytest.mark.parametrize("frozen", [False, True])
def test_zero_support_and_frozen_distributions_keep_parameters(n_components, frozen):
    """Zero support and an explicit freeze take precedence over the covariance floor."""
    initial = GaussianMixture(
        [[-1.0], [1.0]][:n_components],
        [[[0.2]], [[0.4]]][:n_components],
        np.ones(n_components) / n_components,
        frozen=frozen,
    )
    weights = np.ones(4) if frozen else np.zeros(4)

    result = update_emissions(initial, [-2.0, -0.5, 0.5, 2.0], weights, min_covar=10.0)

    assert_emissions_equal(result, initial)
    assert result is not initial


@pytest.mark.parametrize("min_covar", [1e-6, 1e-4])
@pytest.mark.parametrize("operation", ["initialization", "reestimation"])
def test_tiny_positive_covariance_respects_requested_floor(min_covar, operation):
    """Regularize tiny positive definite variances even when Cholesky would succeed."""
    observations = np.array([[-1e-5, -1e-5], [-1e-5, 1e-5], [1e-5, -1e-5], [1e-5, 1e-5]])

    fitted = (
        fit_emissions(observations, min_covar=min_covar)
        if operation == "initialization"
        else update_emissions(
            GaussianMixture([[0.0, 0.0]], [np.eye(2)], [1.0]), observations, np.ones(4), min_covar=min_covar
        )
    )

    assert_allclose(fitted.means[0], [0.0, 0.0], atol=1e-15)
    assert_allclose(fitted.covariances[0], np.eye(2) * min_covar, rtol=1e-12, atol=1e-15)


def test_correlated_rank_deficient_covariance_floors_only_small_eigenvalues():
    """Keep measured variance along the data while bounding its unsupported orthogonal direction."""
    observations = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])

    fitted = fit_emissions(observations)

    assert_allclose(fitted.means[0], [2.0, 4.0])
    assert_allclose(np.linalg.eigvalsh(fitted.covariances[0]), [1e-6, 10.0 / 3.0], rtol=1e-9)


def test_large_offset_retains_small_observed_variances():
    """Estimate variance from centered differences instead of subtracting large second moments."""
    observations = 1e12 + np.array([[-0.25, -0.5], [-0.25, 0.5], [0.25, -0.5], [0.25, 0.5]])

    fitted = fit_emissions(observations)

    assert_allclose(fitted.means[0], [1e12, 1e12], rtol=0, atol=0)
    assert_allclose(fitted.covariances[0], [[0.0625, 0.0], [0.0, 0.25]], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("n_components", [1, 2])
def test_tiny_uniformly_scaled_weights_preserve_weighted_estimate(n_components):
    """Absolute posterior magnitude does not change Gaussian or mixture estimates."""
    initial = GaussianMixture(
        [[-1.0], [1.0]][:n_components], [[[1.0]]] * n_components, np.ones(n_components) / n_components
    )
    observations = np.array([[-2.0], [-0.5], [0.5], [2.0]])
    weights = np.array([0.3, 1.0, 0.4, 0.7])
    expected = update_emissions(initial, observations, weights)

    result = update_emissions(initial, observations, weights * 1e-250)

    assert_allclose(result.means, expected.means, rtol=1e-13)
    assert_allclose(result.covariances, expected.covariances, rtol=1e-13)
    assert_allclose(result.weights, expected.weights, rtol=1e-13)


@pytest.mark.parametrize("direction", [[1e12, 1e12], [638294.7409282152, -292047.4852193635]])
def test_large_covariance_scale_still_returns_representable_positive_definite_matrix(direction):
    """Use a machine-scale floor when the requested absolute floor cannot be represented."""
    vector = np.asarray(direction)
    observations = np.stack([-vector, vector])

    fitted = fit_emissions(observations)
    covariance = np.asarray(fitted.covariances[0])

    assert np.isfinite(np.linalg.cholesky(covariance)).all()
    assert np.linalg.eigvalsh(covariance).min() > 0
    assert_allclose(np.trace(covariance), np.dot(vector, vector), rtol=1e-14)


def test_nearly_degenerate_mixture_updates_keep_bounded_covariance_and_monotone_likelihood():
    """Repeated EM fits nearly collinear clusters without covariance collapse or likelihood jumps."""
    offset = np.linspace(-0.03, 0.03, 20)
    cluster = np.column_stack([offset, 2 * offset + 1e-5 * (-1.0) ** np.arange(len(offset))])
    observations = np.concatenate([cluster + np.array([-1.0, -0.5]), cluster + np.array([1.0, 0.5])])
    distribution = GaussianMixture([[-0.5, -0.2], [0.5, 0.2]], [np.eye(2)] * 2, [0.5, 0.5])
    previous = -np.inf
    for _ in range(40):
        distribution = update_emissions(distribution, observations, np.ones(len(observations)))
        for covariance in distribution.covariances:
            assert np.linalg.eigvalsh(covariance).min() >= 1e-6 * (1 - 1e-9)
        log_scores = np.column_stack(
            [
                multivariate_normal.logpdf(observations, mean, covariance)
                for mean, covariance in zip(distribution.means, distribution.covariances, strict=True)
            ]
        )
        with np.errstate(divide="ignore"):
            log_scores += np.log(distribution.weights)
        likelihood = logsumexp(log_scores, axis=1).sum()
        assert np.isfinite(likelihood)
        assert likelihood >= previous - 1e-8
        previous = likelihood


@pytest.mark.parametrize("n_components", [1, 2])
def test_constant_observations_use_configured_floor_through_initialization(n_components):
    """Propagate the configured variance floor through Gaussian and mixture initialization."""
    result = fit_emissions(np.repeat([[2.0, 3.0]], 5, axis=0), n_components=n_components, min_covar=0.05)
    active = result.weights > 0

    for mean, covariance in zip(result.means[active], result.covariances[active], strict=True):
        assert_allclose(mean, [2.0, 3.0])
        assert_allclose(covariance, np.eye(2) * 0.05, atol=1e-14)


def test_zero_weight_far_observation_does_not_erase_mixture_component_variance():
    """Center each component around supported data when distant samples have zero responsibility."""
    distribution = GaussianMixture([[1e16, 1e16], [0.0, 0.0]], [np.eye(2), np.eye(2)], [0.5, 0.5])
    observations = np.array([[1e16, 1e16], [-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])

    result = update_emissions(distribution, observations, np.ones(len(observations)))

    assert_allclose(result.means[1], [0.0, 0.0], atol=1e-15)
    assert_allclose(result.covariances[1], np.eye(2), atol=1e-15)


def test_separated_mixture_initialization_retains_small_cluster_moments():
    """Fit each component at its own data scale despite a distant first cluster."""
    result = fit_emissions(np.array([[1e16], [1.0], [2.0]]), n_components=2)

    assert_allclose(result.means[1], [1.5], atol=1e-15)
    assert_allclose(result.covariances[1], [[0.25]], atol=1e-15)


def test_emissions_own_read_only_parameters_and_preserve_them_when_cloned():
    """Keep caller mutation and deepcopy from invalidating cached inference parameters."""
    means = np.array([[0.0]])
    emissions = GaussianMixture(means, [[[1.0]]], [1.0])
    means[0, 0] = 100.0
    cloned = copy.deepcopy(emissions)

    assert_allclose(emissions.means, [[0.0]])
    for instance in [emissions, cloned]:
        for array in [instance.means, instance.covariances, instance.weights, instance.component_frozen]:
            assert not array.flags.writeable
    assert not np.shares_memory(emissions.means, cloned.means)


def test_pickled_emissions_restore_read_only_native_parameters():
    """Keep restored emission parameters immutable so cached inference cannot become stale."""
    emissions = GaussianMixture(
        [[0.0], [3.0]], [[[1.0]], [[2.0]]], [0.3, 0.7], frozen=True, component_frozen=[False, True]
    )

    restored = pickle.loads(pickle.dumps(emissions))

    assert_emissions_equal(restored, emissions)
    for array in [restored.means, restored.covariances, restored.weights, restored.component_frozen]:
        assert not array.flags.writeable


def test_zero_centroid_coordinates_are_updated_during_mixture_initialization():
    """Use the true centroid when assigned observations sum to zero."""
    observations = np.array([[1.0], [2.0], [1.25], [-2.25]])

    emissions = fit_emissions(observations, n_components=2)

    assert_allclose(emissions.means[:, 0], [-2.25, 17 / 12], rtol=1e-10, atol=1e-12)
    assert_allclose(emissions.weights, [0.25, 0.75], rtol=1e-10, atol=1e-12)
    assert_allclose(emissions.covariances[:, 0, 0], [1e-6, 13 / 72], rtol=1e-10, atol=1e-12)
