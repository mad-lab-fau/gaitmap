"""SciPy-specific HMM inference helpers."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from gaitmap_mad.stride_segmentation.hmm._state import (
    GaussianEmissionState,
    GaussianMixtureEmissionState,
    HMMState,
)


def log_emission_probabilities(model: HMMState, observations: np.ndarray) -> np.ndarray:
    """Evaluate log-emission probabilities for each state and observation."""
    log_emissions = np.empty((len(observations), len(model.compiled.emissions)), dtype=float)
    for state_idx, emission in enumerate(model.compiled.emissions):
        if isinstance(emission, GaussianEmissionState):
            log_emissions[:, state_idx] = multivariate_normal.logpdf(
                observations,
                mean=emission.mean,
                cov=emission.covariance,
                allow_singular=True,
            )
            continue
        if isinstance(emission, GaussianMixtureEmissionState):
            component_log_probs = np.column_stack(
                [
                    multivariate_normal.logpdf(
                        observations,
                        mean=component.mean,
                        cov=component.covariance,
                        allow_singular=True,
                    )
                    for component in emission.components
                ]
            )
            with np.errstate(divide="ignore"):
                log_weights = np.log(np.asarray(emission.weights, dtype=float))
            log_emissions[:, state_idx] = logsumexp(component_log_probs + log_weights, axis=1)
            continue
        raise TypeError(f"Unsupported serialized emission state `{type(emission).__name__}`.")
    return log_emissions


def viterbi_decode(model: HMMState, log_emissions: np.ndarray) -> np.ndarray:
    """Run Viterbi decoding on the canonical serialized HMM state."""
    end_probs = np.asarray(model.compiled.graph.end_probs, dtype=float)
    use_terminal_state = np.any(end_probs > 0)
    with np.errstate(divide="ignore"):
        transition_log_probs = np.log(np.asarray(model.compiled.graph.transition_probs, dtype=float))
        start_log_probs = np.log(np.asarray(model.compiled.graph.start_probs, dtype=float))
        end_log_probs = np.log(end_probs)

    n_samples, n_states = log_emissions.shape
    dp = np.full((n_samples, n_states), -np.inf, dtype=float)
    pointers = np.zeros((n_samples, n_states), dtype=int)

    dp[0] = start_log_probs + log_emissions[0]
    for sample_idx in range(1, n_samples):
        scores = dp[sample_idx - 1][:, None] + transition_log_probs
        pointers[sample_idx] = np.argmax(scores, axis=0)
        dp[sample_idx] = scores[pointers[sample_idx], np.arange(n_states)] + log_emissions[sample_idx]

    path = np.zeros(n_samples, dtype=int)
    if use_terminal_state:
        path[-1] = int(np.argmax(dp[-1] + end_log_probs))
    else:
        path[-1] = int(np.argmax(dp[-1]))
    for sample_idx in range(n_samples - 1, 0, -1):
        path[sample_idx - 1] = pointers[sample_idx, path[sample_idx]]
    if use_terminal_state:
        return path
    return path[:-1]


def map_decode(model: HMMState, log_emissions: np.ndarray) -> np.ndarray:
    """Run forward-backward MAP decoding on the canonical serialized HMM state."""
    end_probs = np.asarray(model.compiled.graph.end_probs, dtype=float)
    with np.errstate(divide="ignore"):
        transition_log_probs = np.log(np.asarray(model.compiled.graph.transition_probs, dtype=float))
        start_log_probs = np.log(np.asarray(model.compiled.graph.start_probs, dtype=float))
        end_log_probs = np.log(end_probs)

    n_samples, n_states = log_emissions.shape
    forward = np.full((n_samples, n_states), -np.inf, dtype=float)
    backward = np.full((n_samples, n_states), -np.inf, dtype=float)

    forward[0] = start_log_probs + log_emissions[0]
    for sample_idx in range(1, n_samples):
        forward[sample_idx] = log_emissions[sample_idx] + logsumexp(
            forward[sample_idx - 1][:, None] + transition_log_probs,
            axis=0,
        )

    if np.any(end_probs > 0):
        backward[-1] = end_log_probs
    else:
        backward[-1] = 0.0
    for sample_idx in range(n_samples - 2, -1, -1):
        backward[sample_idx] = logsumexp(
            transition_log_probs + log_emissions[sample_idx + 1][None, :] + backward[sample_idx + 1][None, :],
            axis=1,
        )

    posterior = forward + backward
    return np.argmax(posterior, axis=1)
