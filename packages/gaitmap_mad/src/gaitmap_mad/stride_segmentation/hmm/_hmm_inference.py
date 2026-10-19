"""Numba inference for gaitmap's persisted Gaussian hidden Markov models.

The model contains emitting states only. Initial and termination probabilities
are separate numeric vectors, and prediction returns one state per observation.
"""

import json

import numpy as np
from numba import njit

from gaitmap_mad.stride_segmentation.hmm._hmm_distributions import GaussianMixture


@njit(cache=True)
def _logaddexp(left, right):
    if left == -np.inf:
        return right
    if right == -np.inf:
        return left
    maximum = max(left, right)
    return maximum + np.log1p(np.exp(-abs(left - right)))


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
def _forward(scores, incoming, starts, ends):
    n_samples, n_states = scores.shape
    offsets, sources, transitions = incoming
    forward = np.empty_like(scores)
    forward[0] = starts + scores[0]
    for observation in range(1, n_samples):
        for state in range(n_states):
            total = -np.inf
            for edge in range(offsets[state], offsets[state + 1]):
                total = _logaddexp(total, forward[observation - 1, sources[edge]] + transitions[edge])
            forward[observation, state] = total + scores[observation, state]
    likelihood = -np.inf
    for state in range(n_states):
        likelihood = _logaddexp(likelihood, forward[-1, state] + ends[state])
    return forward, likelihood


@njit(cache=True, nogil=True)
def _posteriors(scores, outgoing, ends, forward, likelihood):
    n_samples, n_states = scores.shape
    offsets, targets, transitions = outgoing
    backward = np.empty_like(scores)
    backward[-1] = ends
    for observation in range(n_samples - 2, -1, -1):
        for state in range(n_states):
            total = -np.inf
            for edge in range(offsets[state], offsets[state + 1]):
                following = targets[edge]
                total = _logaddexp(
                    total,
                    transitions[edge] + scores[observation + 1, following] + backward[observation + 1, following],
                )
            backward[observation, state] = total
    return forward + backward - likelihood


@njit(cache=True, nogil=True)
def _traceback(predecessors, final_scores):
    n_samples, n_states = predecessors.shape
    best, last = -np.inf, 0
    for state in range(n_states):
        if final_scores[state] > best:
            best, last = final_scores[state], state
    path = np.empty(n_samples, dtype=np.int64)
    path[-1] = last
    for observation in range(n_samples - 1, 0, -1):
        path[observation - 1] = predecessors[observation, path[observation]]
    return path, best


@njit(cache=True, nogil=True)
def _viterbi(scores, incoming, starts, ends):
    n_samples, n_states = scores.shape
    offsets, sources, transitions = incoming
    dense = len(sources) == n_states * n_states
    predecessors = np.zeros((n_samples, n_states), dtype=np.int64)
    previous_scores = starts + scores[0]
    for observation in range(1, n_samples):
        next_scores = np.full(n_states, -np.inf)
        for state in range(n_states):
            best = -np.inf
            if dense:
                # Full neighborhoods are ordered 0..n_states-1, so skip indirect indexing.
                for previous in range(n_states):
                    candidate = previous_scores[previous] + transitions[state * n_states + previous]
                    if candidate > best:
                        best = candidate
                        predecessors[observation, state] = previous
            else:
                for edge in range(offsets[state], offsets[state + 1]):
                    previous = sources[edge]
                    candidate = previous_scores[previous] + transitions[edge]
                    if candidate > best:
                        best = candidate
                        predecessors[observation, state] = previous
            next_scores[state] = best + scores[observation, state]
        previous_scores = next_scores
    return _traceback(predecessors, previous_scores + ends)


def _log_probabilities(probabilities):
    with np.errstate(divide="ignore"):
        return np.log(probabilities)


def _transition_edges(probabilities):
    """Group nonzero edges by row, preserving ascending neighbor order for ties."""
    rows, neighbors = np.nonzero(probabilities)
    offsets = np.concatenate(([0], np.cumsum(np.bincount(rows, minlength=len(probabilities)))))
    return offsets, neighbors, _log_probabilities(probabilities[rows, neighbors])


def _owned_array(values, dtype=np.float64):
    result = np.array(values, dtype=dtype, copy=True, order="C")
    result.flags.writeable = False
    return result


class _NoValidPathError(ValueError):
    """Distinguish an impossible state path from invalid observation data."""


class HiddenMarkovModel:
    """Gaussian HMM with explicitly represented emitting states.

    Transition probabilities include the probability of termination in their
    row normalization. ``ends=None`` permits termination in any state. Masks
    preserve permitted transitions whose current probability is zero. All
    numeric parameters are owned, read-only arrays.
    """

    __hash__ = None

    def __init__(
        self,
        emissions,
        transitions,
        starts,
        ends=None,
        *,
        name=None,
        state_names=None,
        state_weights=None,
        transition_mask=None,
        start_mask=None,
        end_mask=None,
    ):
        self.name = name
        self.emissions = tuple(
            GaussianMixture(
                emission.means,
                emission.covariances,
                emission.weights,
                frozen=emission.frozen,
                component_frozen=emission.component_frozen,
            )
            for emission in emissions
        )
        self.n_states = len(self.emissions)
        self.n_features = self.emissions[0].n_features
        self.state_names = (
            tuple(state_names) if state_names is not None else tuple(f"s{i}" for i in range(self.n_states))
        )
        self.transitions = _owned_array(transitions)
        self.starts = _owned_array(starts)
        self.ends = None if ends is None else _owned_array(ends)
        self.state_weights = _owned_array(np.ones(self.n_states) if state_weights is None else state_weights)
        self.transition_mask = _owned_array(
            self.transitions > 0 if transition_mask is None else transition_mask, dtype=bool
        )
        self.start_mask = _owned_array(self.starts > 0 if start_mask is None else start_mask, dtype=bool)
        self.end_mask = (
            None if self.ends is None else _owned_array(self.ends > 0 if end_mask is None else end_mask, dtype=bool)
        )
        self._validate_shapes()
        self._incoming = _transition_edges(self.transitions.T)
        self._outgoing = _transition_edges(self.transitions)
        self._log_starts = _log_probabilities(self.starts)
        self._log_ends = np.zeros(self.n_states) if self.ends is None else _log_probabilities(self.ends)
        self._prepare_emissions()

    def _validate_shapes(self):
        if self.transitions.shape != (self.n_states, self.n_states) or self.starts.shape != (self.n_states,):
            raise ValueError("Transition and initial probabilities must match the emitting state count.")
        if self.ends is not None and self.ends.shape != (self.n_states,):
            raise ValueError("Termination probabilities must match the emitting state count.")
        if self.state_weights.shape != (self.n_states,) or len(self.state_names) != self.n_states:
            raise ValueError("State names and weights must match the emitting state count.")
        if self.transition_mask.shape != self.transitions.shape or self.start_mask.shape != self.starts.shape:
            raise ValueError("Topology masks must match their probability arrays.")
        if self.end_mask is not None and self.end_mask.shape != self.ends.shape:
            raise ValueError("The termination mask must match termination probabilities.")
        if any(emission.n_features != self.n_features for emission in self.emissions):
            raise ValueError("All state emissions must have the same number of features.")

    def __eq__(self, other):
        if not isinstance(other, HiddenMarkovModel):
            return NotImplemented
        if self.name != other.name or self.state_names != other.state_names or self.n_states != other.n_states:
            return False
        parameters = ("transitions", "starts", "ends", "state_weights", "transition_mask", "start_mask", "end_mask")
        if any(not np.array_equal(getattr(self, name), getattr(other, name)) for name in parameters):
            return False
        return all(
            left.frozen == right.frozen
            and all(
                np.array_equal(getattr(left, name), getattr(right, name))
                for name in ("means", "covariances", "weights", "component_frozen")
            )
            for left, right in zip(self.emissions, other.emissions)
        )

    def __repr__(self):
        """Summarize the model without printing its parameter arrays."""
        return f"HiddenMarkovModel(n_states={self.n_states}, n_features={self.n_features}, name={self.name!r})"

    def __deepcopy__(self, memo):
        """Rebuild owning read-only parameters and their inference caches when cloning."""
        result = type(self)(**self.__getstate__())
        memo[id(self)] = result
        return result

    def __getstate__(self):
        """Persist and hash native parameters independently of derived inference caches."""
        return {
            "emissions": self.emissions,
            "transitions": self.transitions,
            "starts": self.starts,
            "ends": self.ends,
            "name": self.name,
            "state_names": self.state_names,
            "state_weights": self.state_weights,
            "transition_mask": self.transition_mask,
            "start_mask": self.start_mask,
            "end_mask": self.end_mask,
        }

    def __setstate__(self, state):
        """Rebuild read-only arrays and inference caches after process transfer."""
        self.__init__(**state)

    def _prepare_emissions(self):
        means, whitening, normalizers, weights, offsets = [], [], [], [], [0]
        for emission in self.emissions:
            for mean, covariance, weight in zip(emission.means, emission.covariances, emission.weights):
                cholesky = np.linalg.cholesky(covariance)
                means.append(mean)
                whitening.append(np.linalg.solve(cholesky, np.eye(self.n_features)))
                normalizers.append(-0.5 * self.n_features * np.log(2 * np.pi) - np.log(np.diag(cholesky)).sum())
                weights.append(weight)
            offsets.append(len(means))
        self._means = np.ascontiguousarray(means)
        self._whitening = np.ascontiguousarray(whitening)
        self._normalizers = np.asarray(normalizers)
        self._component_weights = _log_probabilities(np.asarray(weights))
        self._offsets = np.asarray(offsets, dtype=np.int64)
        self._state_weights = _log_probabilities(self.state_weights)

    def fit(
        self,
        sequences,
        labels=None,
        algorithm="baum-welch",
        stop_threshold=1e-9,
        max_iterations=100_000_000,
        verbose=False,
        n_jobs=1,
        min_covar=1e-6,
        update_emissions=True,
    ):
        """Return a fitted model and history, leaving this model unchanged.

        Parameters
        ----------
        sequences : sequence of array-like
            Independent observation sequences, each with shape
            ``(n_observations, n_features)``. One-dimensional arrays are accepted
            for univariate models. Features must match this model's feature order.
            Training starts from this model's existing parameters.
        labels : sequence of array-like or None, default=None
            One integer state sequence per observation sequence, with one label
            per observation and values in ``[0, n_states)``. Required only for
            ``algorithm="labeled"``. Labels are ignored by Baum-Welch and Viterbi
            training; supplying them does not select supervised training or
            initialize the emissions.
        algorithm : {"baum-welch", "viterbi", "labeled"}, default="baum-welch"
            Baum-Welch uses posterior state responsibilities. Viterbi uses the
            current most likely paths. Labeled training uses the supplied paths.
        stop_threshold : float, default=1e-9
            Stop when the improvement in total forward log likelihood is at most
            this value. A nonnegative improvement within the threshold marks
            convergence. A decrease stops without convergence and remains
            visible in the returned history.
        max_iterations : int, default=100_000_000
            Maximum number of parameter updates. Zero returns an unchanged copy
            of the model and a history containing only its initial likelihood.
        verbose : bool, default=False
            Print the likelihood improvement after each update and the total
            improvement when training ends.
        n_jobs : int, default=1
            Number of workers for independent sequence statistics, following
            joblib's worker-count conventions. ``-1`` uses all available CPUs.
            Workers use threads and their results are combined in input order.
        min_covar : float, default=1e-6
            Minimum estimated covariance eigenvalue, in squared feature units.
            Applied only when updating emissions. Explicitly frozen emissions
            retain their original parameters.
        update_emissions : bool, default=True
            Update emission parameters as well as initial, transition and
            termination probabilities. Set to False to keep emissions fixed.
            Existing topology masks remain in effect for probability updates.

        Returns
        -------
        model : HiddenMarkovModel
            Fitted model after the last completed update. The input model remains
            unchanged, including when fitting stops without convergence.
        history : TrainingHistory
            Initial forward log likelihood and the likelihood after every update,
            update durations and convergence status. The final likelihood
            describes the returned model for every training algorithm.

        Raises
        ------
        ValueError
            If Baum-Welch or Viterbi training encounters a sequence with a
            nonfinite likelihood, including an impossible state path. In this
            case fitting raises instead of returning a model and history.

        Notes
        -----
        Labeled training can start from an impossible model with an initial
        likelihood of negative infinity. If its likelihood remains nonfinite
        after an update, fitting stops without convergence and records that
        likelihood in the returned history.
        """
        from gaitmap_mad.stride_segmentation.hmm._hmm_training import fit_hmm  # noqa: PLC0415

        return fit_hmm(
            self,
            sequences,
            labels=labels,
            algorithm=algorithm,
            stop_threshold=stop_threshold,
            max_iterations=max_iterations,
            verbose=verbose,
            n_jobs=n_jobs,
            min_covar=min_covar,
            update_emissions=update_emissions,
        )

    @staticmethod
    def from_dict(data):
        """Load a native or legacy persisted model through the persistence adapter."""
        from gaitmap_mad.stride_segmentation.hmm._hmm_io import load_model  # noqa: PLC0415

        return load_model(data)

    @classmethod
    def from_json(cls, serialized):
        """Load a model from its JSON string."""
        return cls.from_dict(json.loads(serialized))

    def to_dict(self):
        """Serialize model parameters through the persistence adapter."""
        from gaitmap_mad.stride_segmentation.hmm._hmm_io import serialize_model  # noqa: PLC0415

        return serialize_model(self)

    def to_json(self):
        """Serialize the model without inference caches."""
        return json.dumps(self.to_dict())

    def _prepare_observations(self, samples):
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim == 1 and self.n_features == 1:
            samples = samples.reshape(-1, 1)
        if samples.ndim != 2 or samples.shape[1] != self.n_features or len(samples) == 0:
            raise ValueError(f"Expected a nonempty observation matrix with {self.n_features} columns.")
        if not np.isfinite(samples).all():
            raise ValueError("Observations must contain only finite values.")
        return np.ascontiguousarray(samples)

    def _scores(self, samples):
        return _emission_scores(
            self._prepare_observations(samples),
            self._means,
            self._whitening,
            self._normalizers,
            self._component_weights,
            self._offsets,
            self._state_weights,
        )

    def log_probability(self, samples):
        """Return the summed log likelihood of all valid state paths."""
        _, likelihood = _forward(self._scores(samples), self._incoming, self._log_starts, self._log_ends)
        return likelihood

    def predict_log_proba(self, samples):
        """Return posterior log probabilities for each emitting state."""
        scores = self._scores(samples)
        forward, likelihood = _forward(scores, self._incoming, self._log_starts, self._log_ends)
        if not np.isfinite(likelihood):
            raise _NoValidPathError("No valid state path exists for the observations.")
        return _posteriors(scores, self._outgoing, self._log_ends, forward, likelihood)

    def predict_proba(self, samples):
        """Return posterior probabilities for each emitting state."""
        return np.exp(self.predict_log_proba(samples))

    def predict(self, samples, algorithm="viterbi"):
        """Decode observations with Viterbi or posterior maximum probability.

        Ties resolve to the earliest state in model order. Posterior decoding
        need not produce a transition-valid path. Both algorithms return
        exactly one state index per observation.
        """
        if algorithm == "map":
            return np.argmax(self.predict_log_proba(samples), axis=1)
        if algorithm != "viterbi":
            raise ValueError("Supported prediction algorithms are 'viterbi' and 'map'.")
        path, likelihood = _viterbi(self._scores(samples), self._incoming, self._log_starts, self._log_ends)
        if not np.isfinite(likelihood):
            raise _NoValidPathError("No valid state path exists for the observations.")
        return path
