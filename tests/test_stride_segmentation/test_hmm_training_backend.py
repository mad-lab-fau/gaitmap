"""Training checks at the local model fitting interface."""

import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm

from gaitmap.stride_segmentation.hmm import GaussianMixture, HiddenMarkovModel

_SEQUENCES = [np.array([[-2.0], [-1.0], [0.5], [1.5]]), np.array([[-0.5], [1.0], [2.0]])]


def gaussian(mean, covariance, *, frozen=False):
    """Construct a native single-component emission for a worked training example."""
    return GaussianMixture([mean], [covariance], [1.0], frozen=frozen)


def two_state_model():
    """Construct a connected native model for multi-sequence fitting."""
    return HiddenMarkovModel(
        [gaussian([-1.0], [[1.0]]), gaussian([1.0], [[1.0]])],
        [[0.75, 0.25], [0.2, 0.8]],
        [0.6, 0.4],
    )


@pytest.mark.parametrize("budget", [0, 1])
def test_iteration_budget_counts_parameter_updates_and_reports_returned_model(budget):
    """Zero updates preserve parameters and one update produces the Gaussian MLE."""
    model = HiddenMarkovModel([gaussian([0.0], [[4.0]])], [[1.0]], [1.0])
    initial = model.to_dict()
    observations = np.array([-2.0, -1.0, 1.0, 2.0])

    trained, history = model.fit([observations], max_iterations=budget)

    if budget == 0:
        assert trained.to_dict() == initial
    else:
        assert_allclose(trained.emissions[0].means, [[0.0]])
        assert_allclose(trained.emissions[0].covariances, [[[2.5]]])
    assert model.to_dict() == initial
    assert history.n_iterations == budget
    assert len(history.log_likelihoods) == budget + 1
    assert len(history.iteration_durations) == budget
    assert not history.converged
    assert_allclose(history.log_likelihoods[-1], trained.log_probability(observations))


def test_one_finite_mixture_update_matches_gaussian_responsibilities_and_transition_counts():
    """One update uses soft component responsibilities and explicit termination counts."""
    mixture = GaussianMixture([[-1.0], [1.0]], [[[1.0]], [[1.0]]], [0.5, 0.5])
    model = HiddenMarkovModel([mixture], [[0.8]], [1.0], [0.2])
    observations = np.array([-2.0, -0.5, 1.5, 2.5])
    scores = np.column_stack([norm.pdf(observations, loc=-1), norm.pdf(observations, loc=1)])
    responsibilities = scores / scores.sum(axis=1, keepdims=True)
    counts = responsibilities.sum(axis=0)
    expected_means = (responsibilities * observations[:, None]).sum(axis=0) / counts
    expected_variances = (responsibilities * (observations[:, None] - expected_means) ** 2).sum(axis=0) / counts

    trained, history = model.fit([observations], max_iterations=1)

    assert_allclose(trained.transitions, [[0.75]])
    assert_allclose(trained.ends, [0.25])
    assert_allclose(trained.emissions[0].weights, counts / len(observations))
    assert_allclose(trained.emissions[0].means[:, 0], expected_means)
    assert_allclose(trained.emissions[0].covariances[:, 0, 0], expected_variances)
    assert history.n_iterations == 1
    assert_allclose(history.log_likelihoods[-1], trained.log_probability(observations))


def test_convergence_reports_completed_updates_and_stops_without_a_pending_step():
    """An unchanged likelihood ends fitting after the second, fully recorded update."""
    model = HiddenMarkovModel([gaussian([0.0], [[4.0]])], [[0.8]], [1.0], [0.2])
    observations = np.array([-2.0, -1.0, 1.0, 2.0])

    trained, history = model.fit([observations], max_iterations=20)

    assert history.n_iterations == 2
    assert history.converged
    assert len(history.log_likelihoods) == 3
    assert history.improvements[0] > 0
    assert history.improvements[1] == 0
    assert_allclose(history.log_likelihoods[-1], trained.log_probability(observations))


def test_negative_improvement_is_recorded_without_claiming_convergence():
    """An imposed variance floor can reduce likelihood, which must remain visible."""
    model = HiddenMarkovModel([gaussian([0.0], [[0.1]])], [[1.0]], [1.0])
    observations = np.array([-0.1, 0.1])

    trained, history = model.fit([observations], max_iterations=20, min_covar=1.0)

    assert history.n_iterations == 1
    assert not history.converged
    assert history.improvements[0] < 0
    assert_allclose(history.log_likelihoods[-1], trained.log_probability(observations))


def test_labeled_fit_can_learn_currently_zero_permitted_termination():
    """Known labels can give an initially impossible model a valid end transition."""
    model = HiddenMarkovModel([gaussian([0.0], [[1.0]])], [[1.0]], [1.0], [0.0], end_mask=[True])
    observations = np.array([-2.0, -1.0, 1.0, 2.0])

    trained, history = model.fit([observations], labels=[[0, 0, 0, 0]], algorithm="labeled", max_iterations=1)

    assert history.log_likelihoods[0] == -np.inf
    assert np.isfinite(history.log_likelihoods[1])
    assert history.n_iterations == 1
    assert_allclose(trained.transitions, [[0.75]])
    assert_allclose(trained.ends, [0.25])


def test_labeled_fit_stops_without_convergence_when_likelihood_remains_impossible():
    """A zero-weight state cannot recover by updating its Gaussian or transitions."""
    model = HiddenMarkovModel([gaussian([0.0], [[1.0]])], [[1.0]], [1.0], state_weights=[0.0])
    observations = np.array([-1.0, 1.0])

    trained, history = model.fit([observations], labels=[[0, 0]], algorithm="labeled", max_iterations=3)

    assert history.n_iterations == 1
    assert history.log_likelihoods == [-np.inf, -np.inf]
    assert not history.converged
    assert trained.log_probability(observations) == -np.inf


@pytest.mark.parametrize("use_names", [False, True])
def test_labeled_fit_learns_counts_and_gaussian_moments_without_mutating_input(use_names):
    """Known labels recover observed transition frequencies and emission moments."""
    model = HiddenMarkovModel(
        [gaussian([-1.0], [[2.0]]), gaussian([2.0], [[2.0]])], [[0.5, 0.3], [0.3, 0.5]], [0.6, 0.4], [0.2, 0.2]
    )
    initial = model.to_dict()
    sequences = [np.array([[-2.0], [-1.0], [1.0], [2.0]]), np.array([[1.0], [3.0], [2.0]])]

    labels = [[0, 0, 1, 1], [1, 1, 1]]
    if use_names:
        labels = [[f"s{label}" for label in sequence] for sequence in labels]
    trained, history = model.fit(sequences, labels=labels, algorithm="labeled", max_iterations=1)

    assert model.to_dict() == initial
    assert_allclose(trained.starts, [0.5, 0.5])
    assert_allclose(trained.transitions, [[0.5, 0.5], [0.0, 0.6]])
    assert_allclose(trained.ends, [0.0, 0.4])
    assert_allclose(trained.emissions[0].means, [[-1.5]])
    assert_allclose(trained.emissions[0].covariances, [[[0.25]]])
    assert_allclose(trained.emissions[1].means, [[1.8]])
    assert_allclose(trained.emissions[1].covariances, [[[0.56]]])
    assert history.n_iterations == 1


def test_labeled_training_preserves_topology_zero_edges_and_unvisited_states():
    """An existing zero edge can learn while forbidden edges remain absent."""
    initial = HiddenMarkovModel(
        [gaussian([-1.0], [[1.0]]), gaussian([1.0], [[1.0]]), gaussian([10.0], [[2.0]])],
        np.eye(3),
        [1.0, 0.0, 0.0],
        transition_mask=[[True, True, False], [False, True, False], [False, False, True]],
    )
    trained, _ = initial.fit(
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
    distribution = GaussianMixture([[-1.0], [1.0]], [[[1.0]], [[1.0]]], [0.4, 0.6], frozen=freeze_parameters)
    other_distribution = gaussian([2.0], [[1.0]], frozen=freeze_parameters)
    model = HiddenMarkovModel([distribution, other_distribution], [[0.75, 0.25], [0.15, 0.85]], [0.6, 0.4])
    trained, _ = model.fit(_SEQUENCES, max_iterations=2, update_emissions=freeze_parameters)
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
    model = two_state_model()
    labels = [[0, 0, 1, 1], [0, 1, 1]] if algorithm == "labeled" else None
    serial, serial_history = model.fit(_SEQUENCES, labels, algorithm=algorithm, max_iterations=2)
    parallel, parallel_history = model.fit(_SEQUENCES, labels, algorithm=algorithm, max_iterations=2, n_jobs=n_jobs)
    assert parallel.to_dict() == serial.to_dict()
    assert parallel_history.log_likelihoods == serial_history.log_likelihoods
    assert parallel_history.improvements == serial_history.improvements


@pytest.mark.parametrize("algorithm", ["baum-welch", "viterbi"])
def test_unsupervised_training_ignores_initialization_labels(algorithm):
    """Partially labeled initialization cannot enable semi-supervised fitting."""
    model = two_state_model()
    expected, _ = model.fit(_SEQUENCES, algorithm=algorithm, max_iterations=1)
    actual, _ = model.fit(
        _SEQUENCES,
        labels=[None, [0, 0, 0]],
        algorithm=algorithm,
        max_iterations=1,
    )
    assert actual.to_dict() == expected.to_dict()


@pytest.mark.parametrize("algorithm", ["baum-welch", "viterbi"])
def test_impossible_training_sequence_fails_clearly(algorithm):
    """A disconnected end cannot yield invented posterior training statistics."""
    model = HiddenMarkovModel([gaussian([0.0], [[1.0]])] * 2, [[1.0, 0.0], [0.0, 0.5]], [1.0, 0.0], [0.0, 0.5])
    with pytest.raises(ValueError, match="No valid state path"):
        model.fit([np.zeros((3, 1))], algorithm=algorithm, max_iterations=1)


def test_fitted_model_roundtrips_through_json_and_predicts():
    """Fitting returns a new portable model and leaves the initial parameters unchanged."""
    model = two_state_model()
    initial = model.to_dict()
    trained, history = model.fit(_SEQUENCES, max_iterations=1)
    assert trained is not model
    assert model.to_dict() == initial
    assert history.n_iterations == 1
    restored = HiddenMarkovModel.from_json(trained.to_json())
    assert restored.to_dict() == trained.to_dict()
    assert_allclose(restored.predict_proba(_SEQUENCES[0]), trained.predict_proba(_SEQUENCES[0]))


@pytest.mark.parametrize("algorithm", ["baum-welch", "viterbi", "labeled"])
@pytest.mark.parametrize("frozen", [False, True])
def test_training_applies_requested_covariance_floor_except_to_frozen_emissions(algorithm, frozen):
    """Degenerate observations honor the caller's eigenvalue floor during training."""
    distribution = gaussian([0.0, 0.0], [[0.001, 0.0], [0.0, 0.002]], frozen=frozen)
    model = HiddenMarkovModel([distribution], [[1.0]], [1.0])
    samples = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0], [2.0, 2.0]])
    trained, _ = model.fit(
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


@pytest.mark.parametrize("algorithm", ["baum-welch", "viterbi"])
def test_two_state_training_update_matches_exhaustive_paths_for_separate_sequences(algorithm):
    """Complete paths determine soft or hard starts, ends, transitions, and pooled moments."""
    means = np.array([-1.0, 1.0])
    variances = np.array([0.8, 1.3])
    transitions = np.array([[0.6, 0.2], [0.15, 0.55]])
    starts = np.array([0.7, 0.3])
    ends = np.array([0.2, 0.3])
    model = HiddenMarkovModel(
        [gaussian([mean], [[variance]]) for mean, variance in zip(means, variances)],
        transitions,
        starts,
        ends,
    )
    sequences = [np.array([[-1.2], [0.1], [1.4]]), np.array([[-0.7], [-0.4]])]
    start_counts, end_counts = np.zeros(2), np.zeros(2)
    transition_counts = np.zeros((2, 2))
    state_responsibilities = []
    initial_log_likelihood = 0.0
    for observations in sequences:
        densities = norm.pdf(observations, loc=means, scale=np.sqrt(variances))
        paths = list(itertools.product(range(2), repeat=len(observations)))
        probabilities = []
        for path in paths:
            probability = starts[path[0]] * ends[path[-1]]
            for time, state in enumerate(path):
                probability *= densities[time, state]
                if time:
                    probability *= transitions[path[time - 1], state]
            probabilities.append(probability)
        total = sum(probabilities)
        initial_log_likelihood += np.log(total)
        responsibilities = np.zeros((len(observations), 2))
        best_path = paths[np.argmax(probabilities)]
        for path, probability in zip(paths, probabilities):
            posterior = probability / total if algorithm == "baum-welch" else float(path == best_path)
            start_counts[path[0]] += posterior
            end_counts[path[-1]] += posterior
            for time, state in enumerate(path):
                responsibilities[time, state] += posterior
                if time:
                    transition_counts[path[time - 1], state] += posterior
        state_responsibilities.append(responsibilities)

    trained, history = model.fit(sequences, algorithm=algorithm, max_iterations=1)

    # Each sequence supplies its own start/end events; no transition crosses the boundary.
    outgoing_counts = transition_counts.sum(axis=1) + end_counts
    assert_allclose(trained.starts, start_counts / len(sequences), rtol=1e-12, atol=1e-14)
    assert_allclose(trained.transitions, transition_counts / outgoing_counts[:, None], rtol=1e-12, atol=1e-14)
    assert_allclose(trained.ends, end_counts / outgoing_counts, rtol=1e-12, atol=1e-14)
    observations = np.concatenate(sequences)[:, 0]
    responsibilities = np.concatenate(state_responsibilities)
    for state, emission in enumerate(trained.emissions):
        weights = responsibilities[:, state]
        mean = np.average(observations, weights=weights)
        variance = np.average((observations - mean) ** 2, weights=weights)
        assert_allclose(emission.means, [[mean]], rtol=1e-12, atol=1e-14)
        assert_allclose(emission.covariances, [[[variance]]], rtol=1e-12, atol=1e-14)
    assert_allclose(history.log_likelihoods[0], initial_log_likelihood, rtol=1e-13)
    assert history.n_iterations == 1
