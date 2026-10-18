"""Numba inference for gaitmap's persisted Gaussian hidden Markov models.

This deliberately supports only emitting Gaussian states and the two silent start
and end states. Training remains a separate concern. Prediction returns one
emitting-state index per observation, with no silent-state sentinels.
"""

import copy
import json

import numpy as np
from numba import njit


@njit(cache=True)
def _logaddexp(left, right):
    if left == -np.inf:
        return right
    if right == -np.inf:
        return left
    maximum = max(left, right)
    return maximum + np.log1p(np.exp(-abs(left - right)))


@njit(cache=True)
def _emission_scores(samples, means, whitening, normalizers, component_weights, offsets, state_weights):
    n_samples, dimensions = samples.shape
    n_states = len(offsets) - 1
    scores = np.full((n_samples, n_states), -np.inf)
    for observation in range(n_samples):
        for state in range(n_states):
            total = -np.inf
            for component in range(offsets[state], offsets[state + 1]):
                squared_distance = 0.0
                for row in range(dimensions):
                    transformed = 0.0
                    for column in range(dimensions):
                        transformed += whitening[component, row, column] * (
                            samples[observation, column] - means[component, column]
                        )
                    squared_distance += transformed * transformed
                score = normalizers[component] - 0.5 * squared_distance + component_weights[component]
                total = _logaddexp(total, score)
            scores[observation, state] = total + state_weights[state]
    return scores


@njit(cache=True)
def _forward(scores, transitions, starts, ends):
    n_samples, n_states = scores.shape
    forward = np.empty_like(scores)
    forward[0] = starts + scores[0]
    for observation in range(1, n_samples):
        for state in range(n_states):
            total = -np.inf
            for previous in range(n_states):
                total = _logaddexp(total, forward[observation - 1, previous] + transitions[previous, state])
            forward[observation, state] = total + scores[observation, state]
    likelihood = -np.inf
    for state in range(n_states):
        likelihood = _logaddexp(likelihood, forward[-1, state] + ends[state])
    return forward, likelihood


@njit(cache=True)
def _posteriors(scores, transitions, ends, forward, likelihood):
    n_samples, n_states = scores.shape
    backward = np.empty_like(scores)
    backward[-1] = ends
    for observation in range(n_samples - 2, -1, -1):
        for state in range(n_states):
            total = -np.inf
            for following in range(n_states):
                total = _logaddexp(
                    total,
                    transitions[state, following]
                    + scores[observation + 1, following]
                    + backward[observation + 1, following],
                )
            backward[observation, state] = total
    return forward + backward - likelihood


@njit(cache=True)
def _viterbi(scores, transitions, starts, ends):
    n_samples, n_states = scores.shape
    predecessors = np.zeros((n_samples, n_states), dtype=np.int64)
    previous_scores = starts + scores[0]
    for observation in range(1, n_samples):
        next_scores = np.full(n_states, -np.inf)
        for state in range(n_states):
            best = -np.inf
            for previous in range(n_states):
                candidate = previous_scores[previous] + transitions[previous, state]
                if candidate > best:
                    best = candidate
                    predecessors[observation, state] = previous
            next_scores[state] = best + scores[observation, state]
        previous_scores = next_scores
    best = -np.inf
    last = 0
    for state in range(n_states):
        candidate = previous_scores[state] + ends[state]
        if candidate > best:
            best = candidate
            last = state
    path = np.empty(n_samples, dtype=np.int64)
    path[-1] = last
    for observation in range(n_samples - 1, 0, -1):
        path[observation - 1] = predecessors[observation, path[observation]]
    return path, best


def _log_probabilities(probabilities):
    with np.errstate(divide="ignore"):
        return np.log(probabilities)


class _Gaussian:
    def __init__(self, data):
        if data.get("class") != "Distribution" or data.get("name") != "MultivariateGaussianDistribution":
            raise ValueError("Only full-covariance MultivariateGaussianDistribution emissions are supported.")
        self._data = data
        self.name = data["name"]
        self.parameters = copy.deepcopy(data["parameters"])
        self.frozen = data.get("frozen", False)
        self.mean = np.asarray(self.parameters[0], dtype=np.float64)
        covariance = np.asarray(self.parameters[1], dtype=np.float64)
        self.d = len(self.mean)
        if covariance.shape != (self.d, self.d) or not np.isfinite(covariance).all():
            raise ValueError("Gaussian covariance must be a finite square matrix matching its mean.")
        if not np.isfinite(self.mean).all() or not np.allclose(covariance, covariance.T):
            raise ValueError("Gaussian means must be finite and covariance matrices must be symmetric.")
        try:
            cholesky = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ValueError("Gaussian covariance must be positive definite.") from error
        self.whitening = np.linalg.solve(cholesky, np.eye(self.d))
        self.normalizer = -0.5 * self.d * np.log(2 * np.pi) - np.log(np.diag(cholesky)).sum()

    def to_dict(self):
        return copy.deepcopy(self._data)


class _Mixture:
    def __init__(self, data):
        self._data = data
        self.distributions = [_Gaussian(component) for component in data["distributions"]]
        self.weights = np.asarray(data["weights"], dtype=np.float64)
        if len(self.weights) != len(self.distributions) or not len(self.weights):
            raise ValueError("Gaussian mixtures need one weight per component.")
        if not np.isfinite(self.weights).all() or np.any(self.weights < 0) or not np.isclose(self.weights.sum(), 1.0):
            raise ValueError("Serialized Gaussian mixture weights must be nonnegative probabilities summing to one.")
        self.d = self.distributions[0].d

    def to_dict(self):
        return copy.deepcopy(self._data)


class _State:
    def __init__(self, data):
        self._data = data
        self.name = data["name"]
        self.weight = float(data["weight"])
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError("State weights must be finite and nonnegative.")
        distribution = data["distribution"]
        if distribution is None:
            self.distribution = None
        elif distribution.get("class") == "GeneralMixtureModel":
            self.distribution = _Mixture(distribution)
        else:
            self.distribution = _Gaussian(distribution)

    def to_dict(self):
        return copy.deepcopy(self._data)


class HiddenMarkovModel:
    """Inference-only HMM loaded from legacy pomegranate Gaussian model JSON.

    Emitting states retain their serialized order and precede the silent states.
    ``finite`` indicates whether the model has explicit termination edges. A
    model without those edges can terminate in any emitting state.
    """

    __hash__ = None

    def __init__(self, model):
        if model.get("class") != "HiddenMarkovModel":
            raise ValueError("Expected a serialized HiddenMarkovModel.")
        if model.get("distribution ties") or any(edge[4] is not None for edge in model["edges"]):
            raise ValueError("Tied distributions and transitions are not supported by the inference backend.")
        self._model = copy.deepcopy(model)
        self.name = model["name"]
        self.states = [_State(state) for state in self._model["states"]]
        self.start_index = model["start_index"]
        self.end_index = model["end_index"]
        self.silent_start = model["silent_index"]
        n_states = self.silent_start
        if n_states < 1 or {self.start_index, self.end_index} != {n_states, n_states + 1}:
            raise ValueError("Only emitting states followed by the silent start and end states are supported.")
        if len(self.states) != n_states + 2 or any(state.distribution is None for state in self.states[:n_states]):
            raise ValueError("Additional silent states are not supported.")
        if any(state.distribution is not None for state in self.states[n_states:]):
            raise ValueError("The start and end states must be silent.")
        self.start = self.states[self.start_index]
        self.end = self.states[self.end_index]
        self._load_transitions(model)
        self._prepare_emissions()

    def __eq__(self, other):
        if not isinstance(other, HiddenMarkovModel):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def _load_transitions(self, model):
        n_states = self.silent_start
        self._transitions = np.zeros((n_states + 2, n_states + 2), dtype=np.float64)
        for source, target, probability, _, _ in model["edges"]:
            if not 0 <= source < len(self.states) or not 0 <= target < len(self.states):
                raise ValueError("Transition indices must reference existing states.")
            if target == self.start_index or source == self.end_index:
                raise ValueError("Transitions into the start state or out of the end state are not supported.")
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Transition probabilities must be finite values between zero and one.")
            self._transitions[source, target] = probability
        self.finite = any(edge[1] == self.end_index for edge in model["edges"])
        self._log_transitions = _log_probabilities(self._transitions[:n_states, :n_states])
        self._log_starts = _log_probabilities(self._transitions[self.start_index, :n_states])
        self._log_ends = (
            _log_probabilities(self._transitions[:n_states, self.end_index]) if self.finite else np.zeros(n_states)
        )

    def _prepare_emissions(self):
        n_states = self.silent_start
        components, component_weights, offsets = [], [], [0]
        for state in self.states[:n_states]:
            distribution = state.distribution
            if isinstance(distribution, _Mixture):
                components.extend(distribution.distributions)
                component_weights.extend(distribution.weights)
            else:
                components.append(distribution)
                component_weights.append(1.0)
            offsets.append(len(components))
        self.d = components[0].d
        if any(component.d != self.d for component in components):
            raise ValueError("All Gaussian components must have the same number of dimensions.")
        self._means = np.ascontiguousarray([component.mean for component in components])
        self._whitening = np.ascontiguousarray([component.whitening for component in components])
        self._normalizers = np.asarray([component.normalizer for component in components])
        self._component_weights = _log_probabilities(np.asarray(component_weights))
        self._offsets = np.asarray(offsets, dtype=np.int64)
        self._state_weights = _log_probabilities(np.asarray([state.weight for state in self.states[:n_states]]))

    @classmethod
    def from_dict(cls, model):
        """Load the supported subset of pomegranate's serialized HMM format."""
        return cls(model)

    @classmethod
    def from_json(cls, serialized):
        """Load a model from its JSON string."""
        return cls.from_dict(json.loads(serialized))

    def to_dict(self):
        """Return an independent copy of the original persisted model."""
        return copy.deepcopy(self._model)

    def to_json(self):
        """Serialize the model without inference caches."""
        return json.dumps(self.to_dict())

    def dense_transition_matrix(self):
        """Return transition probabilities, including the silent states."""
        return self._transitions.copy()

    def _scores(self, samples):
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim == 1 and self.d == 1:
            samples = samples.reshape(-1, 1)
        if samples.ndim != 2 or samples.shape[1] != self.d or len(samples) == 0:
            raise ValueError(f"Expected a nonempty observation matrix with {self.d} columns.")
        if not np.isfinite(samples).all():
            raise ValueError("Observations must contain only finite values.")
        return _emission_scores(
            np.ascontiguousarray(samples),
            self._means,
            self._whitening,
            self._normalizers,
            self._component_weights,
            self._offsets,
            self._state_weights,
        )

    def log_probability(self, samples):
        """Return the summed log likelihood of all valid state paths."""
        _, likelihood = _forward(self._scores(samples), self._log_transitions, self._log_starts, self._log_ends)
        return likelihood

    def predict_log_proba(self, samples):
        """Return posterior log probabilities for each emitting state."""
        scores = self._scores(samples)
        forward, likelihood = _forward(scores, self._log_transitions, self._log_starts, self._log_ends)
        if not np.isfinite(likelihood):
            raise ValueError("No valid state path exists for the observations.")
        return _posteriors(scores, self._log_transitions, self._log_ends, forward, likelihood)

    def predict_proba(self, samples):
        """Return posterior probabilities for each emitting state."""
        return np.exp(self.predict_log_proba(samples))

    def predict(self, samples, algorithm="viterbi"):
        """Decode observations with Viterbi or posterior maximum probability.

        Ties resolve to the earliest state in serialized state order. Posterior
        decoding need not produce a transition-valid path. Both algorithms
        return exactly one state index per observation.
        """
        if algorithm == "map":
            return np.argmax(self.predict_log_proba(samples), axis=1)
        if algorithm != "viterbi":
            raise ValueError("Supported prediction algorithms are 'viterbi' and 'map'.")
        path, likelihood = _viterbi(self._scores(samples), self._log_transitions, self._log_starts, self._log_ends)
        if not np.isfinite(likelihood):
            raise ValueError("No valid state path exists for the observations.")
        return path
