"""Backend-neutral representation of a fitted hidden Markov model."""

from __future__ import annotations

import numpy as np

from gaitmap.base import _BaseSerializable


class HmmModel(_BaseSerializable):
    """Parameters required to decode a Gaussian-mixture hidden Markov model.

    Parameters are stored as plain NumPy arrays so fitted models can be cloned,
    hashed, and serialized without importing the library used for training.
    """

    transition_probabilities: np.ndarray
    start_probabilities: np.ndarray
    end_probabilities: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    weights: np.ndarray
    n_components: np.ndarray
    data_columns: tuple[str, ...]
    stride_states: tuple[int, ...]

    def __init__(
        self,
        transition_probabilities: np.ndarray,
        start_probabilities: np.ndarray,
        end_probabilities: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
        weights: np.ndarray,
        n_components: np.ndarray,
        data_columns: tuple[str, ...],
        stride_states: tuple[int, ...],
    ) -> None:
        self.transition_probabilities = transition_probabilities
        self.start_probabilities = start_probabilities
        self.end_probabilities = end_probabilities
        self.means = means
        self.covariances = covariances
        self.weights = weights
        self.n_components = n_components
        self.data_columns = data_columns
        self.stride_states = stride_states

    @property
    def n_states(self) -> int:
        """Number of hidden states."""
        return len(self.start_probabilities)

    @classmethod
    def from_legacy_pomegranate(
        cls,
        payload: dict,
        *,
        data_columns: tuple[str, ...],
        stride_states: tuple[int, ...],
    ) -> HmmModel:
        """Compile a pomegranate 0.14 ``HiddenMarkovModel`` dictionary."""
        emission_indices = [index for index, state in enumerate(payload["states"]) if state["distribution"] is not None]
        state_index = {legacy_index: index for index, legacy_index in enumerate(emission_indices)}
        n_states = len(emission_indices)

        components = []
        component_weights = []
        for legacy_index in emission_indices:
            distribution = payload["states"][legacy_index]["distribution"]
            if distribution.get("class") == "GeneralMixtureModel":
                components.append(distribution["distributions"])
                component_weights.append(distribution["weights"])
            else:
                components.append([distribution])
                component_weights.append([1.0])

        n_components = np.array([len(state_components) for state_components in components])
        n_features = len(components[0][0]["parameters"][0])
        means = np.zeros((n_states, int(n_components.max()), n_features))
        covariances = np.zeros((n_states, int(n_components.max()), n_features, n_features))
        weights = np.zeros((n_states, int(n_components.max())))
        for state, (state_components, state_weights) in enumerate(zip(components, component_weights)):
            weights[state, : len(state_weights)] = state_weights
            for component, distribution in enumerate(state_components):
                means[state, component] = distribution["parameters"][0]
                covariances[state, component] = distribution["parameters"][1]

        transitions = np.zeros((n_states, n_states))
        starts = np.zeros(n_states)
        ends = np.zeros(n_states)
        for source, target, probability, *_ in payload["edges"]:
            if source == payload["start_index"] and target in state_index:
                starts[state_index[target]] = probability
            elif target == payload["end_index"] and source in state_index:
                ends[state_index[source]] = probability
            elif source in state_index and target in state_index:
                transitions[state_index[source], state_index[target]] = probability

        return cls(
            transition_probabilities=transitions,
            start_probabilities=starts,
            end_probabilities=ends,
            means=means,
            covariances=covariances,
            weights=weights,
            n_components=n_components,
            data_columns=data_columns,
            stride_states=stride_states,
        )
