"""SciPy implementation of hidden Markov model inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from gaitmap.base import _BaseSerializable
from gaitmap_mad.stride_segmentation.hmm._model import HmmModel


class ScipyHmmInference(_BaseSerializable):
    """Decode fitted HMM parameters without a training-library dependency."""

    def predict(self, model: HmmModel, data: pd.DataFrame) -> np.ndarray:
        """Return the most likely hidden-state sequence using Viterbi decoding."""
        try:
            observations = np.ascontiguousarray(data[list(model.data_columns)].to_numpy())
        except KeyError as e:
            raise ValueError(
                "The provided feature data is expected to have the following columns:\n\n"
                f"{model.data_columns}\n\n"
                "But it only has the following columns:\n\n"
                f"{data.columns}"
            ) from e

        if len(observations) < model.n_states:
            raise ValueError(
                "The provided feature data is expected to have at least as many samples as the number of states "
                f"of the model ({model.n_states}). But it only has {len(observations)} samples."
            )

        return self._viterbi(model, self._log_emission_probabilities(model, observations))

    @staticmethod
    def _log_emission_probabilities(model: HmmModel, observations: np.ndarray) -> np.ndarray:
        log_emissions = np.empty((len(observations), model.n_states), dtype=float)
        for state in range(model.n_states):
            n_components = int(model.n_components[state])
            component_log_probabilities = np.column_stack(
                [
                    multivariate_normal.logpdf(
                        observations,
                        mean=model.means[state, component],
                        cov=model.covariances[state, component],
                        allow_singular=True,
                    )
                    for component in range(n_components)
                ]
            )
            with np.errstate(divide="ignore"):
                log_weights = np.log(model.weights[state, :n_components])
            log_emissions[:, state] = logsumexp(component_log_probabilities + log_weights, axis=1)
        return log_emissions

    @staticmethod
    def _viterbi(model: HmmModel, log_emissions: np.ndarray) -> np.ndarray:
        use_terminal_state = np.any(model.end_probabilities > 0)
        with np.errstate(divide="ignore"):
            transition_log_probabilities = np.log(model.transition_probabilities)
            start_log_probabilities = np.log(model.start_probabilities)
            end_log_probabilities = np.log(model.end_probabilities)

        n_samples, n_states = log_emissions.shape
        scores = np.full((n_samples, n_states), -np.inf, dtype=float)
        previous_states = np.zeros((n_samples, n_states), dtype=int)
        scores[0] = start_log_probabilities + log_emissions[0]

        for sample in range(1, n_samples):
            candidate_scores = scores[sample - 1][:, None] + transition_log_probabilities
            previous_states[sample] = np.argmax(candidate_scores, axis=0)
            scores[sample] = candidate_scores[previous_states[sample], np.arange(n_states)] + log_emissions[sample]

        path = np.zeros(n_samples, dtype=int)
        if use_terminal_state:
            path[-1] = int(np.argmax(scores[-1] + end_log_probabilities))
        else:
            path[-1] = int(np.argmax(scores[-1]))
        for sample in range(n_samples - 1, 0, -1):
            path[sample - 1] = previous_states[sample, path[sample]]
        return path if use_terminal_state else path[:-1]
