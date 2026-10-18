"""Sequence-level training for the local Gaussian hidden Markov model."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time

import numpy as np
from joblib import Parallel, delayed
from numba import njit

from gaitmap_mad.stride_segmentation.hmm._hmm_distributions import update_emissions as update_emission_parameters
from gaitmap_mad.stride_segmentation.hmm._hmm_inference import _forward, _posteriors, _viterbi


@dataclass
class History:
    """Likelihood and timing measurements for completed training epochs."""

    improvements: list[float] = field(default_factory=list)
    total_improvement: list[float] = field(default_factory=list)
    log_probabilities: list[float] = field(default_factory=list)
    epochs: list[int] = field(default_factory=list)
    epoch_start_times: list[float] = field(default_factory=list)
    epoch_end_times: list[float] = field(default_factory=list)
    epoch_durations: list[float] = field(default_factory=list)
    learning_rates: list[None] = field(default_factory=list)
    n_seen_batches: list[int] = field(default_factory=list)
    initial_log_probability: float | None = None

    def _record(self, epoch, likelihood, previous, started, ended):
        improvement = likelihood - previous
        self.improvements.append(improvement)
        total = self.total_improvement[-1] if self.total_improvement else 0.0
        self.total_improvement.append(total + improvement)
        self.log_probabilities.append(likelihood)
        self.epochs.append(epoch)
        self.epoch_start_times.append(started)
        self.epoch_end_times.append(ended)
        self.epoch_durations.append(ended - started)
        self.learning_rates.append(None)
        self.n_seen_batches.append(0)
        return improvement


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
    sequences = [
        sequence.reshape(-1, 1) if sequence.ndim == 1 and model.n_features == 1 else sequence for sequence in sequences
    ]
    if not sequences or any(len(sequence) == 0 for sequence in sequences):
        raise ValueError("Training needs nonempty observation sequences.")
    for sequence in sequences:
        model._scores(sequence)
    return [np.ascontiguousarray(sequence) for sequence in sequences]


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
    """Fit a new model and return its training history without changing ``model``.

    Sequence statistics are independent, run in threads, and reduce in input
    order. The iteration boundary follows pomegranate 0.14: when the iteration
    cap is reached, the last pending parameter update is applied before return.
    Labels are used only by the ``labeled`` algorithm.
    ``min_covar`` sets the minimum covariance eigenvalue for every estimated
    Gaussian. Frozen emission parameters retain their original values.
    ``update_emissions=False`` fits only the transition probabilities.
    """
    if algorithm not in {"labeled", "baum-welch", "viterbi"}:
        raise ValueError("Supported training algorithms are 'baum-welch', 'viterbi' and 'labeled'.")
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative.")
    sequences = _prepare_sequences(model, sequences)
    observations = np.concatenate(sequences)
    paths = _labeled_paths(model, sequences, labels) if algorithm == "labeled" else [None] * len(sequences)
    current = type(model)(
        model.emissions,
        model.transitions,
        model.starts,
        model.ends,
        name=model.name,
        state_names=model.state_names,
        state_weights=model.state_weights,
        transition_mask=model.transition_mask,
        start_mask=model.start_mask,
        end_mask=model.end_mask,
    )
    history = History()
    pending = None
    iteration = 0
    improvement = np.inf
    previous = 0.0
    with Parallel(n_jobs=n_jobs, backend="threading") as parallel:
        while improvement > stop_threshold or iteration < 1:
            started = time()
            if pending is not None:
                current = _maximize(current, observations, *pending, min_covar, update_emissions)
            if iteration >= max_iterations + 1:
                break
            summaries = parallel(
                delayed(_sequence_statistics)(current, sequence, path, algorithm)
                for sequence, path in zip(sequences, paths)
            )
            likelihood = sum(summary[0] for summary in summaries)
            counts = sum((summary[1] for summary in summaries), np.zeros_like(current.transitions))
            starts = sum((summary[2] for summary in summaries), np.zeros(current.n_states))
            ends = sum((summary[3] for summary in summaries), np.zeros(current.n_states))
            weights = np.concatenate([summary[4] for summary in summaries])
            pending = counts, starts, ends, weights
            if iteration == 0:
                history.initial_log_probability = likelihood
            else:
                improvement = history._record(iteration, likelihood, previous, started, time())
                if verbose:
                    print(f"[{iteration}] Improvement: {improvement}")
            iteration += 1
            previous = likelihood
    if verbose:
        print(f"Total Training Improvement: {sum(history.improvements)}")
    return current, history
