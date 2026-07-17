"""Inference and training backends for pomegranate 1.x."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from gaitmap.base import _BaseSerializable
from gaitmap_mad.stride_segmentation.hmm._model import HmmModel


class PomegranateHmmInference(_BaseSerializable):
    """Decode a compiled HMM with the torch-based pomegranate 1.x backend.

    This backend requires Python 3.10 or newer. On Python 3.9 use
    :class:`ScipyHmmInference` or :class:`LegacyPomegranateHmmInference`.
    """

    def predict(self, model: HmmModel, data: pd.DataFrame) -> np.ndarray:
        """Return the most likely hidden-state sequence."""
        _ensure_modern_python()
        try:
            from pomegranate.distributions import Normal  # noqa: PLC0415
            from pomegranate.gmm import GeneralMixtureModel  # noqa: PLC0415
            from pomegranate.hmm import DenseHMM  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - depends on the optional environment
            raise ImportError(
                "PomegranateHmmInference requires pomegranate 1.x. Install gaitmap with the `hmm` extra."
            ) from e

        try:
            observations = np.ascontiguousarray(data.loc[:, model.data_columns].to_numpy(dtype=np.float32))
        except KeyError as e:
            raise ValueError(
                f"Expected feature columns {model.data_columns}, but received {tuple(data.columns)}."
            ) from e

        distributions = []
        for state in range(model.n_states):
            components = [
                Normal(
                    model.means[state, component].astype(np.float32),
                    model.covariances[state, component].astype(np.float32),
                    covariance_type="full",
                    frozen=True,
                )
                for component in range(model.n_components[state])
            ]
            distributions.append(
                GeneralMixtureModel(
                    components,
                    priors=model.weights[state, : model.n_components[state]].astype(np.float32),
                    frozen=True,
                )
            )

        ends = model.end_probabilities.astype(np.float32) if np.any(model.end_probabilities) else None
        runtime_model = DenseHMM(
            distributions=distributions,
            edges=model.transition_probabilities.astype(np.float32),
            starts=model.start_probabilities.astype(np.float32),
            ends=ends,
        )
        return runtime_model.viterbi(observations[None]).detach().cpu().numpy()[0]


class PomegranateHmmTrainer(_BaseSerializable):
    """Fit a labeled Gaussian-mixture HMM with pomegranate 1.x.

    The fitted torch model is compiled immediately into :class:`HmmModel`.
    Consequently, pomegranate and torch are required for training only and do
    not leak into serialized models or inference.

    This backend requires Python 3.10 or newer. HMM training is not available
    through the Python 3.9 legacy backend.

    Parameters
    ----------
    max_iterations
        Maximum pomegranate transition-training iterations.
    stop_threshold
        Minimum likelihood improvement before training stops.
    covariance_regularization
        Non-negative value added to fitted covariance diagonals.
    random_state
        Seed used for deterministic Gaussian-mixture initialization.

    """

    max_iterations: int
    stop_threshold: float
    covariance_regularization: float
    random_state: int

    def __init__(
        self,
        *,
        max_iterations: int = 10,
        stop_threshold: float = 1e-3,
        covariance_regularization: float = 1e-6,
        random_state: int = 0,
    ) -> None:
        self.max_iterations = max_iterations
        self.stop_threshold = stop_threshold
        self.covariance_regularization = covariance_regularization
        self.random_state = random_state

    def fit(
        self,
        topology: HmmModel,
        data_sequences: Sequence[pd.DataFrame],
        state_sequences: Sequence[np.ndarray],
        *,
        train: Literal["all", "transitions"] = "all",
    ) -> HmmModel:
        """Fit one complete HMM topology and return backend-neutral parameters."""
        _ensure_modern_python()
        if train not in ("all", "transitions"):
            raise ValueError("`train` must be either 'all' or 'transitions'.")
        try:
            from pomegranate.distributions import Normal  # noqa: PLC0415
            from pomegranate.gmm import GeneralMixtureModel  # noqa: PLC0415
            from pomegranate.hmm import DenseHMM  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - exercised in environments without the optional dependency
            raise ImportError(
                "PomegranateHmmTrainer requires pomegranate 1.x. Install gaitmap with the `hmm` extra."
            ) from e

        data, labels, columns = self._validate_training_data(data_sequences, state_sequences, topology.n_states)
        if train == "transitions":
            if not topology.is_fitted:
                raise ValueError("Transition-only training requires a fitted HMM model.")
            if columns != topology.data_columns:
                raise ValueError(f"Expected feature columns {topology.data_columns}, but received {columns}.")
            transitions, starts, ends = self._initial_probabilities(labels, topology)
            return topology.with_parameters(
                transition_probabilities=transitions,
                start_probabilities=starts,
                end_probabilities=ends,
                means=topology.means,
                covariances=topology.covariances,
                weights=topology.weights,
                data_columns=topology.data_columns,
            )

        distributions = []
        for state in range(topology.n_states):
            state_data = np.concatenate(
                [sequence[state_labels == state] for sequence, state_labels in zip(data, labels)]
            )
            n_components = int(topology.n_gmm_components[state])
            if len(state_data) < n_components:
                raise ValueError(
                    f"State {state} has {len(state_data)} samples, but {n_components} GMM components were requested."
                )
            fitted_gmm = GaussianMixture(
                n_components=n_components,
                covariance_type="full",
                reg_covar=self.covariance_regularization,
                random_state=self.random_state,
                n_init=1,
            ).fit(state_data)
            normals = [
                Normal(
                    mean.astype(np.float32),
                    covariance.astype(np.float32),
                    covariance_type="full",
                    frozen=True,
                )
                for mean, covariance in zip(fitted_gmm.means_, fitted_gmm.covariances_)
            ]
            distributions.append(
                GeneralMixtureModel(normals, priors=fitted_gmm.weights_.astype(np.float32), frozen=True)
            )

        transitions, starts, ends = self._initial_probabilities(labels, topology)
        model = DenseHMM(
            distributions=distributions,
            edges=transitions,
            starts=starts,
            ends=ends,
            max_iter=self.max_iterations,
            tol=self.stop_threshold,
            random_state=self.random_state,
        )
        priors = [np.eye(topology.n_states, dtype=np.float32)[state_sequence] for state_sequence in labels]
        model.fit([sequence.astype(np.float32) for sequence in data], priors=priors)
        return self._compile_model(model, columns, topology)

    @staticmethod
    def _validate_training_data(
        data_sequences: Sequence[pd.DataFrame], state_sequences: Sequence[np.ndarray], n_states: int
    ) -> tuple[list[np.ndarray], list[np.ndarray], tuple[str, ...]]:
        if not data_sequences or len(data_sequences) != len(state_sequences):
            raise ValueError("Training data and state labels must contain the same non-zero number of sequences.")
        columns = tuple(data_sequences[0].columns)
        data = []
        labels = []
        for sequence_index, (sequence, state_sequence) in enumerate(zip(data_sequences, state_sequences)):
            if tuple(sequence.columns) != columns:
                raise ValueError("All training sequences must have identical feature columns in the same order.")
            state_sequence = np.asarray(state_sequence, dtype=int)
            if len(sequence) == 0 or state_sequence.ndim != 1 or len(state_sequence) == 0:
                raise ValueError("Every training sequence must be non-empty and have one-dimensional state labels.")
            if len(sequence) != len(state_sequence):
                raise ValueError(f"Training sequence {sequence_index} and its state labels have different lengths.")
            data.append(np.ascontiguousarray(sequence.to_numpy(dtype=float)))
            labels.append(np.ascontiguousarray(state_sequence))

        unique_states = np.unique(np.concatenate(labels))
        if not np.array_equal(unique_states, np.arange(n_states)):
            raise ValueError("State labels must cover every topology state using consecutive integers from zero.")
        return data, labels, columns

    @staticmethod
    def _initial_probabilities(
        labels: list[np.ndarray], topology: HmmModel
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        transitions = np.zeros((topology.n_states, topology.n_states), dtype=np.float32)
        starts = np.zeros(topology.n_states, dtype=np.float32)
        ends = np.zeros(topology.n_states, dtype=np.float32)
        for sequence in labels:
            starts[sequence[0]] += 1
            ends[sequence[-1]] += 1
            np.add.at(transitions, (sequence[:-1], sequence[1:]), 1)

        if np.any(transitions[~topology.allowed_transitions]):
            raise ValueError("State labels contain transitions forbidden by the HMM topology.")
        if np.any(starts[~topology.allowed_starts]):
            raise ValueError("State labels start in a state forbidden by the HMM topology.")
        if np.any(ends[~topology.allowed_ends]):
            raise ValueError("State labels end in a state forbidden by the HMM topology.")

        row_totals = transitions.sum(axis=1) + ends
        if np.any(row_totals == 0):
            raise ValueError("Every state must have an outgoing transition or occur at the end of a sequence.")
        return transitions / row_totals[:, None], starts / starts.sum(), ends / row_totals

    @staticmethod
    def _compile_model(model, columns: tuple[str, ...], topology: HmmModel) -> HmmModel:
        def as_numpy(value) -> np.ndarray:
            return value.detach().cpu().numpy().copy()

        n_states = len(model.distributions)
        n_components = np.array([len(distribution.distributions) for distribution in model.distributions])
        max_components = int(n_components.max())
        n_features = len(columns)
        means = np.zeros((n_states, max_components, n_features))
        covariances = np.zeros((n_states, max_components, n_features, n_features))
        weights = np.zeros((n_states, max_components))
        for state, distribution in enumerate(model.distributions):
            weights[state, : n_components[state]] = as_numpy(distribution.priors)
            for component, normal in enumerate(distribution.distributions):
                means[state, component] = as_numpy(normal.means)
                covariances[state, component] = as_numpy(normal.covs)

        return topology.with_parameters(
            transition_probabilities=np.exp(as_numpy(model.edges)),
            start_probabilities=np.exp(as_numpy(model.starts)),
            end_probabilities=np.exp(as_numpy(model.ends)),
            means=means,
            covariances=covariances,
            weights=weights,
            data_columns=columns,
        )


def _ensure_modern_python() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError(
            "The torch-based pomegranate inference and training backends require Python 3.10 or newer. "
            "On Python 3.9, use ScipyHmmInference or LegacyPomegranateHmmInference for inference."
        )
