"""Public training regressions against the legacy Roth training example."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from gaitmap.stride_segmentation.hmm import SimpleHmm

matplotlib.use("Agg")


@pytest.mark.parametrize("n_components", [1, 2])
@pytest.mark.parametrize("max_iterations", [0, 1])
def test_simple_hmm_covariance_floor_survives_training_and_serialization(n_components, max_iterations):
    """A caller-selected variance floor applies during initialization and re-estimation."""
    x = np.arange(20, dtype=float)
    data = [pd.DataFrame({"x": x, "y": 2 * x})]
    model = SimpleHmm(
        n_states=1,
        n_gmm_components=n_components,
        architecture="fully-connected",
        algo_train="labeled",
        min_covar=0.02,
        max_iterations=max_iterations,
        verbose=False,
    ).self_optimize(data, [np.zeros(len(x), dtype=int)])
    restored = SimpleHmm.from_json(model.to_json()).clone()

    assert restored.min_covar == 0.02
    distribution = restored.model.to_dict()["states"][0]["distribution"]
    for component in distribution.get("distributions", [distribution]):
        assert np.linalg.eigvalsh(component["parameters"][1])[0] == pytest.approx(0.02)


@pytest.mark.parametrize("algorithm", ["baum-welch", "viterbi", "labeled"])
@pytest.mark.parametrize("architecture", ["left-right-strict", "left-right-loose", "fully-connected"])
def test_simple_hmm_training_and_serialization_preserve_known_states(algorithm, architecture):
    """Fit variable-length labeled sequences through the public gaitmap interface."""
    rng = np.random.default_rng(31)
    labels = [np.repeat([0, 1], repeats) for repeats in [20, 30]]
    data = [pd.DataFrame(rng.normal(0, 0.3, (len(path), 2)) + (path[:, None] * 10 - 5)) for path in labels]
    model = SimpleHmm(
        n_states=2,
        n_gmm_components=2,
        architecture=architecture,
        algo_train=algorithm,
        max_iterations=2,
        n_jobs=2,
        verbose=False,
    )

    trained, history = model.self_optimize_with_info(data, labels)
    restored = SimpleHmm.from_json(trained.to_json()).clone()

    for observations, expected in zip(data, labels):
        assert_array_equal(trained.predict_hidden_state_sequence(observations), expected)
        assert_array_equal(restored.predict_hidden_state_sequence(observations), expected)
    assert history.epochs


def test_training_example_preserves_legacy_segmentation():
    """Preserve stride boundaries and stride/transition classification after training."""
    from examples.stride_segmentation.segmentation_hmm_training import hmm

    with np.load(Path(__file__).parent / "fixtures" / "hmm_legacy_training_example.npz") as reference:
        for sensor in ["left_sensor", "right_sensor"]:
            # Individual labels vary when regularization prevents nearly singular mixtures from collapsing.
            # Their classification and all stride borders must remain unchanged.
            assert_array_equal(hmm.hidden_state_sequence_[sensor] >= 5, reference[sensor + "_states"] >= 5)
            assert_array_equal(hmm.stride_list_[sensor].to_numpy(), reference[sensor + "_strides"])
            assert_array_equal(hmm.matches_start_end_original_[sensor], reference[sensor + "_original_strides"])
