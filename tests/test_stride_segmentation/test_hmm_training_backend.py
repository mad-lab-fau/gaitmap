"""Training checks at the local model fitting interface."""

import json
from pathlib import Path

import numpy as np
import pytest
from gaitmap_mad.stride_segmentation.hmm._hmm_inference import HiddenMarkovModel
from gaitmap_mad.stride_segmentation.hmm._hmm_training import fit_hmm
from numpy.testing import assert_allclose

from tests.test_stride_segmentation.test_hmm_inference_backend import gaussian, model_dict

_LEGACY = json.loads((Path(__file__).parent / "fixtures" / "hmm_legacy_training.json").read_text())


@pytest.mark.parametrize("use_names", [False, True])
def test_labeled_fit_learns_counts_and_gaussian_moments_without_mutating_input(use_names):
    """Known labels recover observed transition frequencies and emission moments."""
    serialized = model_dict(
        [gaussian([-1.0], [[2.0]]), gaussian([2.0], [[2.0]])], [[0.5, 0.3], [0.3, 0.5]], [0.6, 0.4], [0.2, 0.2]
    )
    model = HiddenMarkovModel.from_dict(serialized)
    initial = model.to_dict()
    sequences = [np.array([[-2.0], [-1.0], [1.0], [2.0]]), np.array([[1.0], [3.0], [2.0]])]

    labels = [[0, 0, 1, 1], [1, 1, 1]]
    if use_names:
        labels = [[f"s{label}" for label in sequence] for sequence in labels]
    trained, history = fit_hmm(model, sequences, labels=labels, algorithm="labeled", max_iterations=1)

    assert model.to_dict() == initial
    assert_allclose(trained.starts, [0.5, 0.5])
    assert_allclose(trained.transitions, [[0.5, 0.5], [0.0, 0.6]])
    assert_allclose(trained.ends, [0.0, 0.4])
    assert_allclose(trained.emissions[0].means, [[-1.5]])
    assert_allclose(trained.emissions[0].covariances, [[[0.25]]])
    assert_allclose(trained.emissions[1].means, [[1.8]])
    assert_allclose(trained.emissions[1].covariances, [[[0.56]]])
    assert history.epochs == [1]


@pytest.mark.parametrize("reference", _LEGACY["references"])
def test_training_matches_legacy_models_and_history(reference):
    """Several EM updates match the old engine in both termination modes."""
    model = HiddenMarkovModel.from_dict(_LEGACY["initial_models"]["finite" if reference["finite"] else "nonfinite"])
    trained, history = fit_hmm(
        model,
        _LEGACY["sequences"],
        algorithm=reference["algorithm"],
        max_iterations=reference["max_iterations"],
        stop_threshold=reference.get("stop_threshold", 1e-9),
    )
    expected = np.asarray(reference["transition_matrix"])
    assert_allclose(trained.transitions, expected[:2, :2], rtol=1e-11, atol=1e-13)
    assert_allclose(trained.starts, expected[2, :2], rtol=1e-11, atol=1e-13)
    if reference["finite"]:
        assert_allclose(trained.ends, expected[:2, 3], rtol=1e-11, atol=1e-13)
    for emission, expected_parameters in zip(trained.emissions, reference["state_parameters"]):
        for actual_parameter, expected_parameter in zip(
            (emission.means[0], emission.covariances[0]), expected_parameters
        ):
            assert_allclose(actual_parameter, expected_parameter, rtol=1e-11, atol=1e-13)
    assert history.epochs == reference["epochs"]
    assert_allclose(history.initial_log_probability, reference["initial_log_probability"], rtol=1e-12)
    assert_allclose(history.log_probabilities, reference["log_probabilities"], rtol=1e-12)
    assert_allclose(history.improvements, reference["improvements"], rtol=1e-10, atol=1e-13)


def test_labeled_training_preserves_topology_zero_edges_and_unvisited_states():
    """An existing zero edge can learn while forbidden edges remain absent."""
    serialized = model_dict(
        [gaussian([-1.0], [[1.0]]), gaussian([1.0], [[1.0]]), gaussian([10.0], [[2.0]])],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [1.0, 0.0, 0.0],
    )
    serialized["edges"].append([0, 1, 0.0, 0.0, None])
    initial = HiddenMarkovModel.from_dict(serialized)
    trained, _ = fit_hmm(
        initial,
        [[-2.0, -1.0, 1.0, 2.0]],
        labels=[[0, 0, 1, 1]],
        algorithm="labeled",
        max_iterations=1,
    )
    assert_allclose(trained.transitions[0], [0.5, 0.5, 0.0])
    assert_allclose(trained.transitions[2], [0.0, 0.0, 1.0])
    assert_allclose(trained.emissions[2].means, [[10.0]])
    assert_allclose(trained.emissions[2].covariances, [[[2.0]]])
    assert np.array_equal(trained.transition_mask, initial.transition_mask)


@pytest.mark.parametrize("freeze_parameters", [False, True])
def test_frozen_gaussian_and_mixture_remain_fixed_while_transitions_learn(freeze_parameters):
    """Freezing distributions includes mixture proportions and component parameters."""
    distribution = {
        "class": "GeneralMixtureModel",
        "distributions": [gaussian([-1.0], [[1.0]]), gaussian([1.0], [[1.0]])],
        "weights": [0.4, 0.6],
        "frozen": freeze_parameters,
    }
    other_distribution = gaussian([2.0], [[1.0]])
    other_distribution["frozen"] = freeze_parameters
    serialized = model_dict([distribution, other_distribution], [[0.75, 0.25], [0.15, 0.85]], [0.6, 0.4])
    model = HiddenMarkovModel.from_dict(serialized)
    trained, _ = fit_hmm(model, _LEGACY["sequences"], max_iterations=2, update_emissions=freeze_parameters)
    for actual, expected in zip(trained.emissions, model.emissions):
        assert actual.frozen == expected.frozen
        assert_allclose(actual.means, expected.means)
        assert_allclose(actual.covariances, expected.covariances)
        assert_allclose(actual.weights, expected.weights)
    assert not np.allclose(trained.transitions, model.transitions)


@pytest.mark.parametrize("algorithm", ["baum-welch", "viterbi", "labeled"])
@pytest.mark.parametrize("n_jobs", [2, -1])
def test_parallel_sequence_reduction_is_deterministic(algorithm, n_jobs):
    """Thread scheduling does not change fitted parameters or likelihood history."""
    model = HiddenMarkovModel.from_dict(_LEGACY["initial_models"]["nonfinite"])
    labels = [[0, 0, 1, 1, 1], [0, 0, 1, 1], [1, 1, 1]] if algorithm == "labeled" else None
    serial, serial_history = fit_hmm(model, _LEGACY["sequences"], labels, algorithm=algorithm, max_iterations=2)
    parallel, parallel_history = fit_hmm(
        model, _LEGACY["sequences"], labels, algorithm=algorithm, max_iterations=2, n_jobs=n_jobs
    )
    assert parallel.to_dict() == serial.to_dict()
    assert parallel_history.log_probabilities == serial_history.log_probabilities
    assert parallel_history.improvements == serial_history.improvements


@pytest.mark.parametrize("algorithm", ["baum-welch", "viterbi"])
def test_unsupervised_training_ignores_initialization_labels(algorithm):
    """Partially labeled initialization cannot enable semi-supervised fitting."""
    model = HiddenMarkovModel.from_dict(_LEGACY["initial_models"]["nonfinite"])
    expected, _ = fit_hmm(model, _LEGACY["sequences"], algorithm=algorithm, max_iterations=1)
    actual, _ = fit_hmm(
        model,
        _LEGACY["sequences"],
        labels=[None, [0, 0, 0, 0], None],
        algorithm=algorithm,
        max_iterations=1,
    )
    assert actual.to_dict() == expected.to_dict()


@pytest.mark.parametrize("algorithm", ["baum-welch", "viterbi"])
def test_impossible_training_sequence_fails_clearly(algorithm):
    """A disconnected end cannot yield invented posterior training statistics."""
    model = HiddenMarkovModel.from_dict(
        model_dict([gaussian([0.0], [[1.0]])] * 2, [[1.0, 0.0], [0.0, 0.5]], [1.0, 0.0], [0.0, 0.5])
    )
    with pytest.raises(ValueError, match="No valid state path"):
        fit_hmm(model, [np.zeros((3, 1))], algorithm=algorithm, max_iterations=1)


def test_fitted_model_roundtrips_through_json_and_predicts():
    """Fitting returns a new portable model and leaves the initial parameters unchanged."""
    model = HiddenMarkovModel.from_dict(_LEGACY["initial_models"]["nonfinite"])
    initial = model.to_dict()
    trained, history = model.fit(_LEGACY["sequences"], max_iterations=1)
    assert trained is not model
    assert model.to_dict() == initial
    assert history.epochs == [1]
    restored = HiddenMarkovModel.from_json(trained.to_json())
    assert restored.to_dict() == trained.to_dict()
    assert_allclose(restored.predict_proba(_LEGACY["sequences"][0]), trained.predict_proba(_LEGACY["sequences"][0]))


@pytest.mark.parametrize("algorithm", ["baum-welch", "viterbi", "labeled"])
@pytest.mark.parametrize("frozen", [False, True])
def test_training_applies_requested_covariance_floor_except_to_frozen_emissions(algorithm, frozen):
    """Degenerate observations honor the caller's eigenvalue floor during training."""
    distribution = gaussian([0.0, 0.0], [[0.001, 0.0], [0.0, 0.002]])
    distribution["frozen"] = frozen
    model = HiddenMarkovModel.from_dict(model_dict([distribution], [[1.0]], [1.0]))
    samples = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0], [2.0, 2.0]])
    trained, _ = fit_hmm(
        model,
        [samples],
        labels=[[0, 0, 0, 0]],
        algorithm=algorithm,
        max_iterations=1,
        min_covar=0.025,
    )
    if frozen:
        assert_allclose(trained.emissions[0].means, model.emissions[0].means)
        assert_allclose(trained.emissions[0].covariances, model.emissions[0].covariances)
    else:
        # The empirical covariance has eigenvalues 0 and 2.5. Flooring only the
        # null direction adds 0.0125 * [[1, -1], [-1, 1]] to its 1.25 entries.
        assert_allclose(trained.emissions[0].means, [[0.5, 0.5]])
        assert_allclose(trained.emissions[0].covariances, [[[1.2625, 1.2375], [1.2375, 1.2625]]])
    assert np.isfinite(trained.log_probability(samples))
