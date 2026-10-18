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
    sequences = [np.array([[-2.0], [-1.0], [1.0], [2.0]]), np.array([[1.0], [3.0], [2.0]])]

    labels = [[0, 0, 1, 1], [1, 1, 1]]
    if use_names:
        labels = [[f"s{label}" for label in sequence] for sequence in labels]
    trained, history = fit_hmm(model, sequences, labels=labels, algorithm="labeled", max_iterations=1)

    assert model.to_dict() == serialized
    assert_allclose(trained.dense_transition_matrix()[2, :2], [0.5, 0.5])
    assert_allclose(trained.dense_transition_matrix()[:2, :2], [[0.5, 0.5], [0.0, 0.6]])
    assert_allclose(trained.dense_transition_matrix()[:2, 3], [0.0, 0.4])
    assert_allclose(trained.states[0].distribution.parameters[0], [-1.5])
    assert_allclose(trained.states[0].distribution.parameters[1], [[0.25]])
    assert_allclose(trained.states[1].distribution.parameters[0], [1.8])
    assert_allclose(trained.states[1].distribution.parameters[1], [[0.56]])
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
    assert_allclose(trained.dense_transition_matrix(), reference["transition_matrix"], rtol=1e-11, atol=1e-13)
    for actual_state, expected_parameters in zip(trained.states[:2], reference["state_parameters"]):
        for actual_parameter, expected_parameter in zip(actual_state.distribution.parameters, expected_parameters):
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
    trained, _ = fit_hmm(
        HiddenMarkovModel.from_dict(serialized),
        [[-2.0, -1.0, 1.0, 2.0]],
        labels=[[0, 0, 1, 1]],
        algorithm="labeled",
        max_iterations=1,
    )
    assert_allclose(trained.dense_transition_matrix()[0, :3], [0.5, 0.5, 0.0])
    assert_allclose(trained.dense_transition_matrix()[2, :3], [0.0, 0.0, 1.0])
    assert trained.states[2].distribution.to_dict() == serialized["states"][2]["distribution"]
    assert {(edge[0], edge[1]) for edge in trained.to_dict()["edges"]} == {
        (edge[0], edge[1]) for edge in serialized["edges"]
    }


def test_frozen_gaussian_and_mixture_remain_fixed_while_transitions_learn():
    """Freezing distributions includes mixture proportions and component parameters."""
    distribution = {
        "class": "GeneralMixtureModel",
        "distributions": [gaussian([-1.0], [[1.0]]), gaussian([1.0], [[1.0]])],
        "weights": [0.4, 0.6],
        "frozen": True,
    }
    other_distribution = gaussian([2.0], [[1.0]])
    other_distribution["frozen"] = True
    serialized = model_dict([distribution, other_distribution], [[0.75, 0.25], [0.15, 0.85]], [0.6, 0.4])
    model = HiddenMarkovModel.from_dict(serialized)
    trained, _ = fit_hmm(model, _LEGACY["sequences"], max_iterations=2)
    assert trained.states[0].distribution.to_dict() == distribution
    assert trained.states[1].distribution.to_dict() == other_distribution
    assert not np.allclose(trained.dense_transition_matrix(), model.dense_transition_matrix())


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
    """The mutable model fitting wrapper leaves a portable inference model."""
    model = HiddenMarkovModel.from_dict(_LEGACY["initial_models"]["nonfinite"])
    trained, history = model.fit(_LEGACY["sequences"], max_iterations=1, return_history=True)
    assert trained is model
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
        assert trained.states[0].distribution.to_dict() == distribution
    else:
        # The empirical covariance has eigenvalues 0 and 2.5. Flooring only the
        # null direction adds 0.0125 * [[1, -1], [-1, 1]] to its 1.25 entries.
        assert_allclose(trained.states[0].distribution.parameters[0], [0.5, 0.5])
        assert_allclose(trained.states[0].distribution.parameters[1], [[1.2625, 1.2375], [1.2375, 1.2625]])
    assert np.isfinite(trained.log_probability(samples))
