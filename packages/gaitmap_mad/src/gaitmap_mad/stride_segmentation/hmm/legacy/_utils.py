"""Legacy pomegranate-specific helper utilities."""

from __future__ import annotations

import json
import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd

try:
    import pomegranate as pg
except ImportError:  # pragma: no cover - exercised in environments without pomegranate
    pg = None
try:
    from pomegranate.hmm import History
except (ImportError, AttributeError):
    History = Any
from tpcp import BaseTpcpObject
from tpcp._hash import custom_hash

from gaitmap_mad.stride_segmentation.hmm import _repr_utils
from gaitmap_mad.stride_segmentation.hmm._repr_utils import is_serialized_hmm_state
from gaitmap_mad.stride_segmentation.hmm._utils import _DataToShortError, cluster_data_by_labels

ShortenedHMMPrint = _repr_utils.ShortenedHMMPrint


def _add_transition(model, a, b, probability, pseudocount, group) -> None:
    pseudocount = pseudocount or probability
    model.graph.add_edge(a, b, probability=probability, pseudocount=pseudocount, group=group)


def _clone_model(orig_model: pg.HiddenMarkovModel, assert_correct: bool = True) -> pg.HiddenMarkovModel:
    """Clone a legacy pomegranate HMM without changing its values."""
    d = json.loads(orig_model.to_json())
    model = pg.HiddenMarkovModel(str(d["name"]))

    with np.errstate(divide="ignore"):
        states = [pg.State.from_dict(j) for j in d["states"]]

    for cloned_state, state in zip(states, orig_model.states):
        assert cloned_state.name == state.name
        if state.distribution is not None:
            cloned_state.distribution.frozen = state.distribution.frozen
        if isinstance(state.distribution, pg.GeneralMixtureModel):
            cloned_state.distribution.weights[:] = np.copy(state.distribution.weights)

    for i, j in d["distribution ties"]:
        states[i].tie(states[j])

    model.add_states(states)
    model.start = states[d["start_index"]]
    model.end = states[d["end_index"]]

    new_state_order = [state.name for state in states]
    for start, end, data in list(orig_model.graph.edges(data=True)):
        _add_transition(
            model,
            states[new_state_order.index(start.name)],
            states[new_state_order.index(end.name)],
            data["probability"],
            data["pseudocount"],
            data["group"],
        )

    model.bake(verbose=False)

    if assert_correct:
        assert custom_hash(model) == custom_hash(orig_model), (
            "Cloning the provided HMM model failed! Please open an issue on github with an example."
        )

    return model


class _HackyClonableHMMFix(BaseTpcpObject):
    """Mixin that teaches `tpcp.clone` how to clone legacy pomegranate HMMs."""

    @classmethod
    def __clone_param__(cls, param_name: str, value: Any) -> Any:
        legacy_hmm = getattr(pg, "HiddenMarkovModel", None) if pg is not None else None
        if legacy_hmm is not None and isinstance(value, legacy_hmm):
            return _clone_model(value)
        if is_serialized_hmm_state(value):
            return type(value).from_json(value.to_json())
        return super().__clone_param__(param_name, value)


def gmms_from_samples(
    data,
    labels,
    n_components: int,
    n_expected_states: int,
    verbose: bool = False,
    n_init: int = 5,
    n_jobs: int = 1,
):
    """Create Gaussian mixture distributions from clustered samples."""
    if not np.array(data, dtype=object).data.c_contiguous:
        raise ValueError("Memory Layout of given input data is not contiguous! Consider using numpy.ascontiguousarray.")

    clustered_data = cluster_data_by_labels(data, labels)

    if len(clustered_data) < n_expected_states:
        raise ValueError(
            f"The training labels did only provide samples for {len(clustered_data)} states, but "
            f"{n_expected_states} states were expected. "
            "Ensure that the training data contains samples for all states."
        )

    if data[0].ndim == 1:
        clustered_data = [np.reshape(data, (len(data), 1)) for data in clustered_data]

    pg_dist_type = pg.MultivariateGaussianDistribution

    if n_components > 1:
        for cluster in clustered_data:
            if len(cluster) < n_components:
                raise ValueError(
                    f"The training labels did only provide a small number of samples ({len(cluster)}) for one of the "
                    "states. "
                    f"To initialize {n_components} components in a mixture model, we need at least {n_components} "
                    f"samples! "
                    "Ensure that the training data contains enough samples."
                )
        distributions = []
        for cluster in clustered_data:
            dist = pg.GeneralMixtureModel.from_samples(
                pg_dist_type,
                n_components=n_components,
                X=cluster,
                verbose=verbose,
                n_jobs=n_jobs,
                n_init=n_init,
                init="first-k",
            )
            for distribution in dist.distributions:
                if np.any([np.isnan(p).any() for p in distribution.parameters]).any():
                    raise ValueError(
                        "NaN in parameters during distribution fitting! "
                        "This usually happens when there is not enough data for a large number of distributions and "
                        "states. "
                        "To avoid this issue, reduce the number of distributions per state or the number of states. "
                        "Or ideally, provide more data."
                    )
            distributions.append(dist)
    else:
        distributions = [pg_dist_type.from_samples(dataset) for dataset in clustered_data]

    return distributions, clustered_data


def fix_model_names(model):
    """Fix legacy pomegranate state-name ordering for state indices >= 10."""
    for state in model.states:
        if state.name[0] == "s":
            try:
                state_number = int(state.name[1:])
            except ValueError:
                continue
            if state_number >= 10:
                state.name = "s" + chr(87 + state_number)
    return model


def _iter_nested_distributions(distribution):
    if isinstance(distribution, pg.GeneralMixtureModel):
        yield distribution
        for nested_distribution in distribution.distributions:
            yield from _iter_nested_distributions(nested_distribution)
    else:
        yield distribution


def model_params_are_finite(model: pg.HiddenMarkovModel) -> bool:
    """Check if all legacy pomegranate distribution parameters are finite."""
    for state in model.states:
        for distribution in _iter_nested_distributions(state.distribution):
            if hasattr(distribution, "parameters"):
                for param in distribution.parameters:
                    if not np.all(np.isfinite(param)):
                        return False
            if hasattr(distribution, "weights") and not np.all(np.isfinite(distribution.weights)):
                return False
    return True


def check_history_for_training_failure(history: History) -> None:
    """Warn if the legacy pomegranate training history indicates failure."""
    if not np.all(np.isfinite(history.improvements)) or np.any(np.array(history.improvements) < 0):
        warnings.warn(
            "During training the improvement per epoch became NaN/infinite or negative! "
            "Run `self_optimize_with_info` and inspect the history element for more information. "
            "With a high likelihood, the final model is not usable and will result in errors during prediction. "
            "This usually happens when there is not enough data for a large number of distributions and "
            "states. "
            "To avoid this issue, reduce the number of distributions per state or the number of states. "
            "Or ideally, provide more data."
        )


def get_state_by_name(model: pg.HiddenMarkovModel, state_name: str):
    """Get a state object by name."""
    for state in model.states:
        if state.name == state_name:
            return state
    raise ValueError(f"State {state_name} not found within given model.")


def add_transition(model: pg.HiddenMarkovModel, transition: tuple[str, str], transition_probability: float) -> None:
    """Add a transition to an existing model by state names."""
    model.add_transition(
        get_state_by_name(model, transition[0]),
        get_state_by_name(model, transition[1]),
        transition_probability,
    )


def get_model_distributions(model: pg.HiddenMarkovModel) -> list[pg.Distribution]:
    """Return all emitting distributions from a legacy pomegranate model."""
    distributions = []
    for state in model.states:
        if state.distribution is not None:
            distributions.append(state.distribution)
    return distributions


def labels_to_strings(labelsequence: list[np.ndarray | None]) -> list[list[str] | None]:
    """Convert integer label sequences to legacy state names."""
    assert isinstance(labelsequence, list), "labelsequence must be list!"

    labelsequence_str = []
    for sequence in labelsequence:
        if sequence is None:
            labelsequence_str.append(sequence)
            continue
        labelsequence_str.append([f"s{i:02}" for i in sequence])
    return labelsequence_str


def predict(
    model: Any | None,
    data: pd.DataFrame,
    *,
    expected_columns: tuple[str, ...],
    algorithm: Literal["viterbi", "map"],
) -> np.ndarray:
    """Predict a hidden-state sequence with a legacy pomegranate runtime model."""
    if model is None:
        raise ValueError(
            "You need to train the HMM before calling `predict_hidden_state_sequence`. "
            "Use `self_optimize` or `self_optimize_with_info` for that."
        )

    try:
        data = data[list(expected_columns)]
    except KeyError as e:
        raise ValueError(
            "The provided feature data is expected to have the following columns:\n\n"
            f"{expected_columns}\n\n"
            "But it only has the following columns:\n\n"
            f"{data.columns}"
        ) from e

    if len(data) < len(model.states) - 2:
        raise _DataToShortError(
            "The provided feature data is expected to have at least as many samples as the number of states "
            f"of the model ({len(model.states) - 2}). "
            f"But it only has {len(data)} samples."
        )

    data = np.ascontiguousarray(data.to_numpy())
    try:
        labels_predicted = np.asarray(model.predict(data.copy(), algorithm=algorithm))
    except Exception as e:
        if not model_params_are_finite(model):
            raise ValueError(
                "Prediction failed! (See error above.). "
                "However, the provided pomegranate model has non-finite/NaN parameters. "
                "This might be the source of the observed error and indicates problems during training. "
                "Check the training history and the model parameters to confirm invalid training behaviour. "
                "Unfortunately, there is no way to automatically fix these issues. "
                "Simply speaking, your training data could not be represented well by the selected model architecture. "
                "Check for obvious errors in your pre-processing or try to use a different model architecture. "
            ) from e
        raise ValueError(
            "Prediction failed! (See error above.). "
            "Unfortunately, we are not sure what happened. "
            "The error was caused by pomegrante internals."
        ) from e

    if algorithm == "viterbi":
        labels_predicted = labels_predicted[1:-1]
    return np.asarray(labels_predicted)
