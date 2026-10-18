"""Sequence-level training for the local Gaussian hidden Markov model."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from joblib import Parallel, delayed
from numba import njit

from gaitmap_mad.stride_segmentation.hmm._hmm_distributions import update_emissions as update_emission_parameters
from gaitmap_mad.stride_segmentation.hmm._hmm_inference import _forward, _posteriors, _viterbi


@dataclass
class TrainingHistory:
    """Forward marginal likelihoods of the initial model and every completed update.

    Scores use the same marginal objective for every training algorithm. Hard
    and labeled updates can decrease it without making the model invalid.
    """

    log_likelihoods: list[float] = field(default_factory=list)
    iteration_durations: list[float] = field(default_factory=list)
    converged: bool = False

    @property
    def improvements(self):
        """Return the likelihood improvement for each completed update."""
        return np.diff(self.log_likelihoods).tolist()

    @property
    def n_iterations(self):
        """Return the number of completed parameter updates."""
        return len(self.iteration_durations)


@njit(cache=True, nogil=True)
def _path_statistics(path, n_states, has_end):
    weights = np.zeros((len(path), n_states))
    transitions = np.zeros((n_states, n_states))
    starts = np.zeros(n_states)
    ends = np.zeros(n_states)
    starts[path[0]] = 1.0
    for observation, state in enumerate(path):
        weights[observation, state] = 1.0
        if observation:
            transitions[path[observation - 1], state] += 1.0
    if has_end:
        ends[path[-1]] = 1.0
    return transitions, starts, ends, weights


def _labeled_paths(model, sequences, labels):
    if labels is None or len(labels) != len(sequences):
        raise ValueError("Labeled training needs one label sequence per observation sequence.")
    names = {name: index for index, name in enumerate(model.state_names)}
    paths = []
    for sequence, sequence_labels in zip(sequences, labels):
        if sequence_labels is None or len(sequence_labels) != len(sequence):
            raise ValueError("Labeled training needs exactly one emitting-state label per observation.")
        path = np.asarray(
            [names[label] if isinstance(label, str) else label for label in sequence_labels], dtype=np.int64
        )
        if np.any(path < 0) or np.any(path >= model.n_states):
            raise ValueError("Labels must identify an emitting state.")
        if not model.start_mask[path[0]] or not np.all(model.transition_mask[path[:-1], path[1:]]):
            raise ValueError("Labels contain a transition outside the model topology.")
        if model.ends is not None and not model.end_mask[path[-1]]:
            raise ValueError("Labels end in a state without a permitted termination transition.")
        paths.append(path)
    return paths


@njit(cache=True, nogil=True)
def _expected_transitions(forward, posterior, scores, transitions, has_end):
    n_samples, n_states = scores.shape
    counts = np.zeros((n_states, n_states))
    starts = np.exp(posterior[0])
    ends = np.exp(posterior[-1]) if has_end else np.zeros(n_states)
    for observation in range(n_samples - 1):
        for following in range(n_states):
            if not np.isfinite(forward[observation + 1, following]):
                continue
            remainder = (
                scores[observation + 1, following]
                + posterior[observation + 1, following]
                - forward[observation + 1, following]
            )
            for state in range(n_states):
                if transitions[state, following] == -np.inf:
                    continue
                counts[state, following] += np.exp(
                    forward[observation, state] + transitions[state, following] + remainder
                )
    return counts, starts, ends


def _sequence_statistics(model, sequence, path, algorithm):
    scores = model._scores(sequence)
    forward, likelihood = _forward(scores, model._log_transitions, model._log_starts, model._log_ends)
    if algorithm == "labeled":
        counts, starts, ends, weights = _path_statistics(path, model.n_states, model.ends is not None)
    elif not np.isfinite(likelihood):
        raise ValueError("No valid state path exists for a training sequence.")
    elif algorithm == "viterbi":
        path, _ = _viterbi(scores, model._log_transitions, model._log_starts, model._log_ends)
        counts, starts, ends, weights = _path_statistics(path, model.n_states, model.ends is not None)
    else:
        posterior = _posteriors(scores, model._log_transitions, model._log_ends, forward, likelihood)
        counts, starts, ends = _expected_transitions(
            forward, posterior, scores, model._log_transitions, model.ends is not None
        )
        weights = np.exp(posterior)
    return likelihood, counts, starts, ends, weights


def _maximize(model, observations, counts, starts, ends, weights, min_covar, update_emissions):
    counts = counts * model.transition_mask
    starts = starts * model.start_mask
    ends = ends * model.end_mask if model.ends is not None else np.zeros(model.n_states)
    totals = counts.sum(axis=1) + ends
    observed = totals > 0
    transitions = model.transitions.copy()
    transitions[observed] = counts[observed] / totals[observed, None]
    initial = starts / starts.sum() if starts.sum() > 0 else model.starts
    termination = None if model.ends is None else model.ends.copy()
    if termination is not None:
        termination[observed] = ends[observed] / totals[observed]
    emissions = (
        tuple(
            update_emission_parameters(emission, observations, state_weights, min_covar=min_covar)
            for emission, state_weights in zip(model.emissions, weights.T)
        )
        if update_emissions
        else model.emissions
    )
    return type(model)(
        emissions,
        transitions,
        initial,
        termination,
        name=model.name,
        state_names=model.state_names,
        state_weights=model.state_weights,
        transition_mask=model.transition_mask,
        start_mask=model.start_mask,
        end_mask=model.end_mask,
    )


def _prepare_sequences(model, sequences):
    sequences = [np.asarray(sequence, dtype=np.float64) for sequence in sequences]
    if not sequences or any(len(sequence) == 0 for sequence in sequences):
        raise ValueError("Training needs nonempty observation sequences.")
    return [model._prepare_observations(sequence) for sequence in sequences]


def _summarize_sequences(model, sequences, paths, algorithm, parallel):
    summaries = parallel(
        delayed(_sequence_statistics)(model, sequence, path, algorithm) for sequence, path in zip(sequences, paths)
    )
    likelihood = sum(summary[0] for summary in summaries)
    counts = sum((summary[1] for summary in summaries), np.zeros_like(model.transitions))
    starts = sum((summary[2] for summary in summaries), np.zeros(model.n_states))
    ends = sum((summary[3] for summary in summaries), np.zeros(model.n_states))
    weights = np.concatenate([summary[4] for summary in summaries])
    return likelihood, (counts, starts, ends, weights)


def fit_hmm(
    model,
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
    """Return a new fitted model and a history containing its final likelihood.

    ``max_iterations`` limits completed parameter updates. Zero leaves the
    initial parameters unchanged. Sequence statistics run independently in
    threads and reduce in input order. Labels apply only to ``labeled`` fitting.
    ``min_covar`` bounds estimated covariance eigenvalues. Frozen emissions
    retain their parameters, and ``update_emissions=False`` fits transitions only.
    Recorded scores and stopping use forward marginal likelihood for every
    algorithm. Hard or labeled updates can stop on a decrease without implying
    an invalid model.
    A nonfinite likelihood after an update stops fitting without convergence.
    """
    if algorithm not in {"labeled", "baum-welch", "viterbi"}:
        raise ValueError("Supported training algorithms are 'baum-welch', 'viterbi' and 'labeled'.")
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative.")
    sequences = _prepare_sequences(model, sequences)
    observations = np.concatenate(sequences)
    paths = _labeled_paths(model, sequences, labels) if algorithm == "labeled" else [None] * len(sequences)
    current = deepcopy(model)
    with Parallel(n_jobs=n_jobs, backend="threading") as parallel:
        likelihood, statistics = _summarize_sequences(current, sequences, paths, algorithm, parallel)
        history = TrainingHistory(log_likelihoods=[likelihood])
        while history.n_iterations < max_iterations:
            started = perf_counter()
            current = _maximize(current, observations, *statistics, min_covar, update_emissions)
            likelihood, statistics = _summarize_sequences(current, sequences, paths, algorithm, parallel)
            improvement = likelihood - history.log_likelihoods[-1]
            history.log_likelihoods.append(likelihood)
            history.iteration_durations.append(perf_counter() - started)
            if verbose:
                print(f"[{history.n_iterations}] Improvement: {improvement}")
            if not np.isfinite(likelihood):
                break
            if improvement <= stop_threshold:
                history.converged = improvement >= 0
                break
    if verbose:
        print(f"Total Training Improvement: {sum(history.improvements)}")
    return current, history
