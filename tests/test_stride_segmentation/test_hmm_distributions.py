"""Emission updates agree with independently captured legacy Gaussian and GMM fits."""

import json
from pathlib import Path

import numpy as np
import pytest
from gaitmap_mad.stride_segmentation.hmm._hmm_distributions import fit_distribution, update_distribution
from numpy.testing import assert_allclose
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "hmm_distributions_legacy.json").read_text())


def assert_distribution_close(actual, expected):
    """Compare numerical distribution parameters without freezing incidental JSON metadata."""
    if expected["class"] == "GeneralMixtureModel":
        assert_allclose(actual["weights"], expected["weights"], rtol=1e-9, atol=1e-11)
        for component, reference in zip(actual["distributions"], expected["distributions"]):
            assert_distribution_close(component, reference)
    else:
        assert_allclose(actual["parameters"][0], expected["parameters"][0], rtol=1e-9, atol=1e-11)
        assert_allclose(actual["parameters"][1], expected["parameters"][1], rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("case", REFERENCE["updates"], ids=lambda case: case["name"])
def test_weighted_reestimation_matches_legacy(case):
    """Match weighted MLE and soft component responsibilities over consecutive updates."""
    distribution = case["initial"]
    original = json.dumps(distribution, sort_keys=True)
    for expected in case["updates"]:
        distribution = update_distribution(
            distribution, np.asarray(case["observations"]), np.asarray(case["sample_weights"])
        )
        assert_distribution_close(distribution, expected)
    assert json.dumps(case["initial"], sort_keys=True) == original


@pytest.mark.parametrize(
    "case",
    REFERENCE["initializations"],
    ids=lambda case: f"{len(case['observations'][0])}d-{case['n_components']}components",
)
def test_deterministic_initialization_matches_legacy(case):
    """Match first-k initialization followed by EM at the legacy stopping threshold."""
    result = fit_distribution(np.asarray(case["observations"]), n_components=case["n_components"])

    assert_distribution_close(result, case["expected"])


@pytest.mark.parametrize(
    "case",
    [case for case in REFERENCE["support_and_repair"] if case["name"] in {"zero_component", "frozen_component"}],
    ids=lambda case: case["name"],
)
def test_component_zero_support_and_freezes_match_legacy(case):
    """Keep supported mixture semantics while replacing legacy covariance and small-weight quirks."""
    result = update_distribution(case["initial"], np.asarray(case["observations"]), np.asarray(case["sample_weights"]))

    assert_distribution_close(result, case["expected"])


@pytest.mark.parametrize("case", REFERENCE["updates"], ids=lambda case: case["name"])
@pytest.mark.parametrize("frozen", [False, True])
def test_zero_support_and_frozen_distributions_keep_parameters(case, frozen):
    """Do not change unsupported state emissions or explicitly frozen whole distributions."""
    initial = json.loads(json.dumps(case["initial"]))
    initial["frozen"] = frozen
    weights = np.asarray(case["sample_weights"]) if frozen else np.zeros(len(case["observations"]))

    result = update_distribution(initial, np.asarray(case["observations"]), weights, min_covar=10.0)

    assert result == initial
    assert result is not initial


@pytest.mark.parametrize("min_covar", [1e-6, 1e-4])
@pytest.mark.parametrize("operation", ["initialization", "reestimation"])
def test_tiny_positive_covariance_respects_requested_floor(min_covar, operation):
    """Regularize tiny positive definite variances even when Cholesky would succeed."""
    observations = np.array([[-1e-5, -1e-5], [-1e-5, 1e-5], [1e-5, -1e-5], [1e-5, 1e-5]])

    fitted = (
        fit_distribution(observations, min_covar=min_covar)
        if operation == "initialization"
        else update_distribution(REFERENCE["updates"][0]["initial"], observations, np.ones(4), min_covar=min_covar)
    )

    assert_allclose(fitted["parameters"][0], [0.0, 0.0], atol=1e-15)
    assert_allclose(fitted["parameters"][1], np.eye(2) * min_covar, rtol=1e-12, atol=1e-15)


def test_correlated_rank_deficient_covariance_floors_only_small_eigenvalues():
    """Keep measured variance along the data while bounding its unsupported orthogonal direction."""
    observations = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])

    fitted = fit_distribution(observations)

    assert_allclose(fitted["parameters"][0], [2.0, 4.0])
    assert_allclose(np.linalg.eigvalsh(fitted["parameters"][1]), [1e-6, 10.0 / 3.0], rtol=1e-9)


def test_large_offset_retains_small_observed_variances():
    """Estimate variance from centered differences instead of subtracting large second moments."""
    observations = 1e12 + np.array([[-0.25, -0.5], [-0.25, 0.5], [0.25, -0.5], [0.25, 0.5]])

    fitted = fit_distribution(observations)

    assert_allclose(fitted["parameters"][0], [1e12, 1e12], rtol=0, atol=0)
    assert_allclose(fitted["parameters"][1], [[0.0625, 0.0], [0.0, 0.25]], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("case", REFERENCE["updates"][:2], ids=lambda case: case["name"])
def test_tiny_uniformly_scaled_weights_preserve_weighted_estimate(case):
    """Absolute posterior magnitude does not change the fitted state or mixture."""
    observations = np.asarray(case["observations"])
    weights = np.asarray(case["sample_weights"])

    result = update_distribution(case["initial"], observations, weights * 1e-250)

    assert_distribution_close(result, case["updates"][0])


@pytest.mark.parametrize("direction", [[1e12, 1e12], [638294.7409282152, -292047.4852193635]])
def test_large_covariance_scale_still_returns_representable_positive_definite_matrix(direction):
    """Use a machine-scale floor when the requested absolute floor cannot be represented."""
    vector = np.asarray(direction)
    observations = np.stack([-vector, vector])

    fitted = fit_distribution(observations)
    covariance = np.asarray(fitted["parameters"][1])

    assert np.isfinite(np.linalg.cholesky(covariance)).all()
    assert np.linalg.eigvalsh(covariance).min() > 0
    assert_allclose(np.trace(covariance), np.dot(vector, vector), rtol=1e-14)


def test_real_gait_mixture_updates_keep_bounded_covariance_and_monotone_likelihood():
    """Repeated updates avoid the legacy covariance collapse and large negative repair jumps."""
    case = REFERENCE["updates"][2]
    observations = np.asarray(case["observations"])
    distribution = case["initial"]
    previous = -np.inf
    for _ in range(40):
        distribution = update_distribution(distribution, observations, np.ones(len(observations)))
        components = distribution["distributions"]
        for component in components:
            assert np.linalg.eigvalsh(component["parameters"][1]).min() >= 1e-6 * (1 - 1e-9)
        log_scores = np.column_stack(
            [multivariate_normal.logpdf(observations, *component["parameters"]) for component in components]
        )
        with np.errstate(divide="ignore"):
            log_scores += np.log(distribution["weights"])
        likelihood = logsumexp(log_scores, axis=1).sum()
        assert np.isfinite(likelihood)
        assert likelihood >= previous - 1e-8
        previous = likelihood


@pytest.mark.parametrize("n_components", [1, 2])
def test_constant_observations_use_configured_floor_through_initialization(n_components):
    """Propagate the configured variance floor through Gaussian and mixture initialization."""
    result = fit_distribution(np.repeat([[2.0, 3.0]], 5, axis=0), n_components=n_components, min_covar=0.05)
    active = (
        [component for component, weight in zip(result["distributions"], result["weights"]) if weight > 0]
        if n_components > 1
        else [result]
    )

    for component in active:
        assert_allclose(component["parameters"][0], [2.0, 3.0])
        assert_allclose(component["parameters"][1], np.eye(2) * 0.05, atol=1e-14)
