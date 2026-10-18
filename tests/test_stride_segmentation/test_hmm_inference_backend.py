"""Inference agrees with independent Gaussian densities and exhaustive HMM paths."""

import copy
import itertools
import pickle

import numpy as np
import pytest
from gaitmap_mad.stride_segmentation.hmm._hmm_distributions import GaussianMixture
from gaitmap_mad.stride_segmentation.hmm._hmm_inference import HiddenMarkovModel
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import multivariate_normal
from tpcp._hash import custom_hash


def gaussian(mean, covariance):
    """Create a legacy full-covariance Gaussian definition."""
    return {
        "class": "Distribution",
        "name": "MultivariateGaussianDistribution",
        "parameters": [mean, covariance],
        "frozen": False,
    }


def model_dict(distributions, transitions, starts, ends=None, weights=None):
    """Create a legacy model with separate emitting and silent states."""
    n_states = len(distributions)
    states = [
        {"class": "State", "name": f"s{i}", "distribution": distribution, "weight": weight}
        for i, (distribution, weight) in enumerate(zip(distributions, weights or [1.0] * n_states))
    ]
    states.extend({"class": "State", "name": name, "distribution": None, "weight": 1.0} for name in ["start", "end"])
    edges = [
        [i, j, probability, probability, None]
        for i, row in enumerate(transitions)
        for j, probability in enumerate(row)
        if probability
    ]
    edges.extend([n_states, i, probability, probability, None] for i, probability in enumerate(starts) if probability)
    if ends is not None:
        edges.extend(
            [i, n_states + 1, probability, probability, None] for i, probability in enumerate(ends) if probability
        )
    return {
        "class": "HiddenMarkovModel",
        "name": "example",
        "states": states,
        "start": states[-2],
        "end": states[-1],
        "start_index": n_states,
        "end_index": n_states + 1,
        "silent_index": n_states,
        "edges": edges,
        "distribution ties": [],
    }


@pytest.mark.parametrize("layout", ["C", "F"])
def test_full_covariance_gaussian_sequence_likelihood(layout):
    """Gaussian scores are independent of the input memory layout."""
    mean = [0.5, -1.0]
    covariance = [[2.0, 0.8], [0.8, 0.9]]
    samples = np.array([[1.1, -0.3], [-1.0, 0.4], [0.5, -1.0]], order=layout)
    model = HiddenMarkovModel.from_dict(model_dict([gaussian(mean, covariance)], [[0.8]], [1.0], [0.2]))

    expected = np.sum(multivariate_normal.logpdf(samples, mean=mean, cov=covariance)) + 2 * np.log(0.8) + np.log(0.2)
    assert_allclose(model.log_probability(samples), expected, rtol=1e-13)
    assert_array_equal(model.predict(samples), [0, 0, 0])
    assert_allclose(model.predict_proba(samples), np.ones((3, 1)))


@pytest.mark.parametrize("finite", [False, True])
def test_mixture_inference_matches_exhaustive_paths(finite):
    """Path enumeration verifies likelihoods, the best path and state posteriors."""
    mixture = {
        "class": "GeneralMixtureModel",
        "distributions": [gaussian([-1.0], [[0.4]]), gaussian([1.0], [[1.2]])],
        "weights": [0.3, 0.7],
    }
    distributions = [mixture, gaussian([2.0], [[0.7]])]
    transitions = [[0.65, 0.25], [0.15, 0.55]] if finite else [[0.75, 0.25], [0.2, 0.8]]
    starts = [0.6, 0.4]
    ends = [0.1, 0.3] if finite else None
    weights = [0.7, 1.3]
    samples = np.array([[-0.8], [0.4], [1.7], [-0.3]])
    serialized = model_dict(distributions, transitions, starts, ends, weights)
    model = HiddenMarkovModel.from_dict(serialized)

    # Enumerating complete paths gives an independent reference for both dynamic programs.
    density = np.column_stack(
        [
            0.3 * multivariate_normal.pdf(samples, mean=[-1.0], cov=[[0.4]])
            + 0.7 * multivariate_normal.pdf(samples, mean=[1.0], cov=[[1.2]]),
            multivariate_normal.pdf(samples, mean=[2.0], cov=[[0.7]]),
        ]
    )
    paths = list(itertools.product(range(2), repeat=len(samples)))
    path_probabilities = []
    for path in paths:
        probability = starts[path[0]]
        for observation, state in enumerate(path):
            probability *= density[observation, state] * weights[state]
            if observation:
                probability *= transitions[path[observation - 1]][state]
        if ends is not None:
            probability *= ends[path[-1]]
        path_probabilities.append(probability)
    likelihood = sum(path_probabilities)
    posterior = np.zeros((len(samples), 2))
    for path, probability in zip(paths, path_probabilities):
        for observation, state in enumerate(path):
            posterior[observation, state] += probability / likelihood

    assert (model.ends is not None) == finite
    assert_allclose(model.log_probability(samples), np.log(likelihood), rtol=1e-13)
    assert_array_equal(model.predict(samples), paths[np.argmax(path_probabilities)])
    assert_allclose(model.predict_proba(samples), posterior, rtol=1e-13)
    assert_allclose(model.predict_log_proba(samples), np.log(posterior), rtol=1e-13)
    assert_array_equal(model.predict(samples, algorithm="map"), np.argmax(posterior, axis=1))


def test_decoding_impossible_path_raises_without_inventing_a_state_path():
    """A disconnected end state gives no decode and a zero sequence probability."""
    model = HiddenMarkovModel.from_dict(
        model_dict([gaussian([0.0], [[1.0]])] * 2, [[1.0, 0.0], [0.0, 0.5]], [1.0, 0.0], [0.0, 0.5])
    )
    samples = np.zeros((3, 1))
    assert model.log_probability(samples) == -np.inf
    for algorithm in ["viterbi", "map"]:
        with pytest.raises(ValueError, match="No valid state path"):
            model.predict(samples, algorithm=algorithm)
    with pytest.raises(ValueError, match="No valid state path"):
        model.predict_proba(samples)


def test_ties_use_serialized_state_order_and_single_observation_is_supported():
    """Equal probabilities choose the first state without silent-state labels."""
    model = HiddenMarkovModel.from_dict(model_dict([gaussian([0.0], [[1.0]])] * 2, [[0.5, 0.5]] * 2, [0.5, 0.5]))
    for samples in [np.array([1.0]), np.zeros(4)]:
        assert_array_equal(model.predict(samples), np.zeros(len(samples), dtype=int))
        assert_array_equal(model.predict(samples, algorithm="map"), np.zeros(len(samples), dtype=int))


def test_inference_serialization_does_not_mutate_or_alias_persisted_model():
    """Callers can clone and serialize a model before or after inference."""
    serialized = model_dict([gaussian([0.0], [[1.0]])], [[0.8]], [1.0], [0.2])
    model = HiddenMarkovModel.from_dict(serialized)
    reference = copy.deepcopy(model.to_dict())
    serialized["states"][0]["name"] = "changed by caller"
    model.predict([0.0, 1.0])
    model.predict_proba([0.0, 1.0])
    assert model.to_dict() == reference
    assert model.state_names[0] == "s0"
    assert_allclose(model.emissions[0].means, [[0.0]])
    assert_allclose(model.emissions[0].covariances, [[[1.0]]])
    assert_allclose(model.transitions, [[0.8]])
    assert_allclose(model.starts, [1.0])
    assert_allclose(model.ends, [0.2])
    returned = model.to_dict()
    returned["state_names"][0] = "changed returned dict"
    assert model.to_dict() == reference
    assert HiddenMarkovModel.from_json(model.to_json()).to_dict() == reference


def test_numeric_parameters_are_owned_so_caller_changes_cannot_stale_inference_caches():
    """Mutating constructor inputs cannot change a constructed model's probabilities."""
    means = np.array([[0.0]])
    covariance = np.array([[[1.0]]])
    transitions, starts, ends = np.array([[0.8]]), np.array([1.0]), np.array([0.2])
    emission = GaussianMixture(means, covariance, [1.0])
    model = HiddenMarkovModel([emission], transitions, starts, ends)
    expected = model.log_probability([0.0, 1.0])

    means[:] = 10.0
    covariance[:] = 100.0
    transitions[:] = 0.2
    starts[:] = 0.5
    ends[:] = 0.8

    for candidate in [model, copy.deepcopy(model)]:
        assert_allclose(candidate.log_probability([0.0, 1.0]), expected)
        for parameter in [candidate.transitions, candidate.starts, candidate.ends, candidate.emissions[0].means]:
            with pytest.raises(ValueError, match="read-only"):
                parameter.flat[0] = 0.0


def test_parameter_hash_is_independent_of_numeric_cache_reconstruction():
    """Cloning keeps parameter identity when derived scores differ in their final bit."""
    model = HiddenMarkovModel.from_dict(model_dict([gaussian([0.0], [[1.0]])], [[0.8]], [1.0], [0.2]))
    reconstructed = copy.deepcopy(model)
    # Rebuilding a floating point cache may change its final bit without any
    # parameter change. Such caches must not participate in parameter hashing.
    reconstructed._normalizers = np.nextafter(reconstructed._normalizers, np.inf)

    assert reconstructed == model
    assert_allclose(reconstructed.log_probability([0.0, 1.0]), model.log_probability([0.0, 1.0]))
    assert custom_hash(reconstructed) == custom_hash(model)


def test_pickle_restores_read_only_parameters_and_matching_inference():
    """Worker-process transfer rebuilds safe arrays and inference caches from parameters."""
    model = HiddenMarkovModel.from_dict(model_dict([gaussian([0.0], [[1.0]])], [[0.8]], [1.0], [0.2]))
    restored = pickle.loads(pickle.dumps(model))

    assert custom_hash(restored) == custom_hash(model)
    assert_allclose(restored.log_probability([0.0, 1.0]), model.log_probability([0.0, 1.0]))
    for parameter in [restored.transitions, restored.starts, restored.ends, restored.emissions[0].means]:
        with pytest.raises(ValueError, match="read-only"):
            parameter.flat[0] = 0.0


@pytest.mark.parametrize("unsupported", ["distribution", "silent state", "distribution ties", "transition ties"])
def test_unsupported_persisted_model_features_fail_clearly(unsupported):
    """Unsupported legacy model types do not silently change inference semantics."""
    serialized = model_dict([gaussian([0.0], [[1.0]])], [[0.8]], [1.0], [0.2])
    if unsupported == "distribution":
        serialized["states"][0]["distribution"]["name"] = "NormalDistribution"
    elif unsupported == "silent state":
        serialized["states"][0]["distribution"] = None
    elif unsupported == "distribution ties":
        serialized["distribution ties"] = [[0, 0]]
    else:
        serialized["edges"][0][4] = "group"
    with pytest.raises(ValueError, match="supported"):
        HiddenMarkovModel.from_dict(serialized)
