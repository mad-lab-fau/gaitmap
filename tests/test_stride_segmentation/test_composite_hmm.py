"""Composition learns state transitions from complete, labeled model instances."""

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal

from gaitmap.stride_segmentation.hmm import CompositeHmm, GaussianMixture, HiddenMarkovModel


def test_composition_counts_repeated_instances_and_separate_record_boundaries():
    """Adjacent instances of the same model retain their last-to-first connection."""
    rest = HiddenMarkovModel([GaussianMixture([[-5.0]], [[[0.25]]], [1.0])], [[0.8]], [1.0], [0.2])
    stride = HiddenMarkovModel(
        [GaussianMixture([[mean]], [[[0.25]]], [1.0]) for mean in [1.0, 5.0]],
        [[0.6, 0.4], [0.0, 0.4]],
        [1.0, 0.0],
        [0.0, 0.6],
    )
    sequences = [np.array([-5.0, 1.0, 5.0, 1.0, 5.0, -5.0, -5.0])[:, None], np.array([1.0, 5.0, -5.0])[:, None]]
    labels = [
        pd.DataFrame(
            [(5, 7, "rest"), (3, 5, "stride"), (1, 3, "stride"), (0, 1, "rest")], columns=["start", "end", "label"]
        ),
        pd.DataFrame([(0, 2, "stride"), (2, 3, "rest")], columns=["start", "end", "label"]),
    ]
    composition = CompositeHmm({"rest": rest, "stride": stride})

    trained, history = composition.fit_transitions(sequences, labels)

    # Pooled paths are [0,1,2,1,2,0,0] and [1,2,0]. No transition joins records.
    assert_allclose(trained.model.starts, [0.5, 0.5, 0.0])
    assert_allclose(trained.model.transitions, [[0.5, 0.5, 0.0], [0.0, 0.0, 1.0], [2 / 3, 1 / 3, 0.0]])
    for actual, expected in zip(trained.model.emissions, rest.emissions + stride.emissions, strict=True):
        assert_array_equal(actual.means, expected.means)
        assert_array_equal(actual.covariances, expected.covariances)
        assert_array_equal(actual.weights, expected.weights)
    assert history.n_iterations == 1
    assert composition.model is None

    expected_labels = ["rest", "stride", "stride", "stride", "stride", "rest", "rest"]
    assert_array_equal(trained.predict_labels(sequences[0]), expected_labels)
    assert_array_equal(trained.predict_hidden_state_sequence(sequences[0]), [0, 1, 2, 1, 2, 0, 0])
    # Saving and cloning must preserve the named interpretation of internal states.
    restored = CompositeHmm.from_json(trained.to_json()).clone()
    assert_array_equal(restored.predict_labels(sequences[0]), expected_labels)
    assert_allclose(restored.model.log_probability(sequences[0]), trained.model.log_probability(sequences[0]))
