"""Quarantined pomegranate 0.14 inference adapter for Python 3.9."""

import sys

import numpy as np
import pandas as pd

from gaitmap.base import _BaseSerializable
from gaitmap_mad.stride_segmentation.hmm._model import HmmModel


class LegacyPomegranateHmmInference(_BaseSerializable):
    """Decode a compiled model with pomegranate 0.14 on Python 3.9."""

    def predict(self, model: HmmModel, data: pd.DataFrame) -> np.ndarray:
        """Return the legacy pomegranate Viterbi state sequence."""
        if sys.version_info >= (3, 10):
            raise RuntimeError("The legacy pomegranate backend is only supported on Python 3.9.")
        try:
            import pomegranate as pg  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - depends on the optional environment
            raise ImportError("Install gaitmap with the `hmm` extra to use the legacy backend.") from e

        observations = np.ascontiguousarray(data.loc[:, model.data_columns].to_numpy(dtype=float))
        distributions = []
        for state in range(model.n_states):
            components = [
                pg.MultivariateGaussianDistribution(model.means[state, component], model.covariances[state, component])
                for component in range(model.n_components[state])
            ]
            with np.errstate(divide="ignore"):
                distributions.append(
                    pg.GeneralMixtureModel(components, weights=model.weights[state, : model.n_components[state]])
                )

        state_names = [f"s{state}" if state < 10 else f"s{chr(87 + state)}" for state in range(model.n_states)]
        runtime_model = pg.HiddenMarkovModel.from_matrix(
            model.transition_probabilities,
            distributions,
            model.start_probabilities,
            ends=model.end_probabilities if np.any(model.end_probabilities) else None,
            state_names=state_names,
        )
        return np.asarray(runtime_model.predict(observations, algorithm="viterbi"))[1:-1]
