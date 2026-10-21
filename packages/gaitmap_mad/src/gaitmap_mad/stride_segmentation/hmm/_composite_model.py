"""Composition of trained HMM submodels from labeled model instances."""

from collections.abc import Mapping, Sequence
from typing import Self

import numpy as np
import pandas as pd
from tpcp import BaseTpcpObject

from gaitmap_mad.stride_segmentation.hmm._hmm_inference import HiddenMarkovModel, _NoValidPathError
from gaitmap_mad.stride_segmentation.hmm._hmm_training import TrainingHistory
from gaitmap_mad.stride_segmentation.hmm._utils import extract_transitions_starts_stops_from_hidden_state_sequence


def _initialize_composite(submodels, sequences, labels, *, algorithm="viterbi", fallback_state=None):
    """Align labeled instances and join trained blocks with their observed connections."""
    offsets = np.r_[0, np.cumsum([model.n_states for model in submodels.values()])]
    state_offsets = dict(zip(submodels, offsets[:-1], strict=True))
    transitions = np.zeros((offsets[-1], offsets[-1]))
    emissions, state_weights = [], []
    for model, start, end in zip(submodels.values(), offsets[:-1], offsets[1:], strict=True):
        transitions[start:end, start:end] = model.transitions
        emissions.extend(model.emissions)
        state_weights.extend(model.state_weights)
    # Submodel termination is replaced by connections in the complete sequence.
    row_totals = transitions.sum(axis=1, keepdims=True)
    np.divide(transitions, row_totals, out=transitions, where=row_totals > 0)

    paths = []
    for sequence, regions in zip(sequences, labels, strict=True):
        path = np.zeros(len(sequence), dtype=np.int64)
        for start, end, label in regions[["start", "end", "label"]].itertuples(index=False, name=None):
            if end == start:
                continue
            model = submodels[label]
            try:
                path[start:end] = model.predict(sequence[start:end], algorithm=algorithm) + state_offsets[label]
            except _NoValidPathError:
                if fallback_state is None or end - start >= model.n_states:
                    raise
                # Roth retains its background fallback for undecodable short regions.
                path[start:end] = fallback_state
        paths.append(path)

    observed, starts, _ = extract_transitions_starts_stops_from_hidden_state_sequence(paths)
    initial = np.zeros(offsets[-1])
    initial[starts] = 1.0 / len(starts)
    for source, target in sorted(observed):
        if transitions[source, target] == 0:
            transitions[source, target] = 0.1
    row_totals = transitions.sum(axis=1, keepdims=True)
    np.divide(transitions, row_totals, out=transitions, where=row_totals > 0)
    return HiddenMarkovModel(emissions, transitions, initial, state_weights=state_weights), paths


class CompositeHmm(BaseTpcpObject):
    """Combine named, trained HMMs using annotated model instances.

    Each annotation table has ``start``, exclusive ``end``, and ``label`` columns.
    Labels name the corresponding submodel. Non-overlapping intervals must cover
    each complete observation sequence. Separate rows preserve consecutive
    instances of the same submodel, such as back-to-back strides.

    Submodels must use the same feature columns in the same order. Their start
    and termination probabilities determine the alignment within each interval.
    ``fit_transitions`` estimates complete-sequence transitions and initial-state
    probabilities from these alignments, keeping all emissions fixed. Predictions
    may end in any internal state, allowing recordings to stop mid-instance.

    Parameters
    ----------
    submodels
        Trained numeric models keyed by the names used in the annotation tables.
        Mapping order determines internal state indices.
    model
        Fitted combined model, populated in the instance returned by fitting.

    """

    def __init__(self, submodels: Mapping[str, HiddenMarkovModel], model: HiddenMarkovModel | None = None):
        self.submodels = submodels
        self.model = model

    @property
    def state_groups(self) -> dict[str, list[int]]:
        """Return the internal state indices belonging to each named submodel."""
        groups, offset = {}, 0
        for name, model in self.submodels.items():
            groups[name] = list(range(offset, offset + model.n_states))
            offset += model.n_states
        return groups

    def fit_transitions(
        self, sequences: Sequence[np.ndarray], labels: Sequence[pd.DataFrame]
    ) -> tuple[Self, TrainingHistory]:
        """Return a fitted composition and history without changing its submodels.

        ``sequences`` contains feature matrices. Each corresponding ``labels``
        table covers the sequence with labeled model instances, using sample
        positions in that matrix. Internal states are inferred per instance;
        their observed transition counts determine the combined probabilities.
        No transitions are counted across observation-sequence boundaries.
        """
        observations = [np.asarray(sequence) for sequence in sequences]
        model, paths = _initialize_composite(self.submodels, observations, labels)
        model, history = model.fit(observations, paths, algorithm="labeled", max_iterations=1, update_emissions=False)
        return type(self)(self.submodels, model), history

    def predict_hidden_state_sequence(self, features: np.ndarray, algorithm="viterbi") -> np.ndarray:
        """Return one internal state index per feature observation."""
        return self.model.predict(features, algorithm=algorithm)

    def predict_labels(self, features: np.ndarray, algorithm="viterbi") -> np.ndarray:
        """Return one submodel name per feature observation."""
        names = np.concatenate([np.repeat(name, model.n_states) for name, model in self.submodels.items()])
        return names[self.predict_hidden_state_sequence(features, algorithm=algorithm)]
