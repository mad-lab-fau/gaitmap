"""Emission updates agree with independently captured legacy Gaussian and GMM fits."""

import json
from pathlib import Path

import numpy as np
import pytest
from gaitmap_mad.stride_segmentation.hmm._hmm_distributions import fit_distribution, update_distribution
from numpy.testing import assert_allclose

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


@pytest.mark.parametrize("case", REFERENCE["support_and_repair"], ids=lambda case: case["name"])
def test_covariance_repair_and_component_support_match_legacy(case):
    """Preserve legacy covariance repair, negligible support, and frozen components."""
    result = update_distribution(case["initial"], np.asarray(case["observations"]), np.asarray(case["sample_weights"]))

    if case["name"] == "rank_deficient":
        # Different BLAS versions can either factor this borderline covariance
        # or add the documented 1e-8 repair. Both represent the same MLE.
        assert_allclose(result["parameters"][0], case["expected"]["parameters"][0], atol=1e-11)
        assert_allclose(result["parameters"][1], case["expected"]["parameters"][1], atol=1.1e-8)
        assert np.linalg.eigvalsh(result["parameters"][1]).min() >= -1e-12
    else:
        assert_distribution_close(result, case["expected"])


@pytest.mark.parametrize("case", REFERENCE["updates"], ids=lambda case: case["name"])
@pytest.mark.parametrize("frozen", [False, True])
def test_zero_support_and_frozen_distributions_keep_parameters(case, frozen):
    """Do not change unsupported state emissions or explicitly frozen whole distributions."""
    initial = json.loads(json.dumps(case["initial"]))
    initial["frozen"] = frozen
    weights = np.asarray(case["sample_weights"]) if frozen else np.zeros(len(case["observations"]))

    result = update_distribution(initial, np.asarray(case["observations"]), weights)

    assert result == initial
    assert result is not initial
