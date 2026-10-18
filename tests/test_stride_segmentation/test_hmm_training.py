"""Public training regressions against the legacy Roth training example."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from gaitmap.stride_segmentation.hmm import RothHmmFeatureTransformer, RothSegmentationHmm, SimpleHmm
from gaitmap.utils.consts import BF_COLS

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
    for covariance in restored.model.emissions[0].covariances:
        assert np.linalg.eigvalsh(covariance)[0] == pytest.approx(0.02)


@pytest.mark.parametrize("ends_at_record_boundary", [False, True])
def test_roth_training_combines_stride_states_with_only_termination_transitions(ends_at_record_boundary):
    """Minimum-length strides can end without a self-loop and still combine with transitions."""
    data = pd.DataFrame(np.random.default_rng(4).normal(size=(30, 6)), columns=BF_COLS)
    strides = (
        pd.DataFrame({"start": [28], "end": [30]})
        if ends_at_record_boundary
        else pd.DataFrame({"start": [4, 12, 20], "end": [6, 14, 22]})
    )
    model = RothSegmentationHmm(
        stride_model=SimpleHmm(n_states=2, n_gmm_components=1, algo_train="labeled", verbose=False),
        transition_model=SimpleHmm(
            n_states=1, n_gmm_components=1, architecture="fully-connected", algo_train="labeled", verbose=False
        ),
        feature_transform=RothHmmFeatureTransformer(
            sampling_rate_feature_space_hz=100, low_pass_filter=None, axes=["gyr_ml"], features=["raw", "gradient"]
        ),
        max_iterations=1,
        verbose=False,
    ).self_optimize([data], [strides], sampling_rate_hz=100)

    matrix = model.model.transitions
    assert np.isfinite(matrix).all()
    if ends_at_record_boundary:
        assert_array_equal(matrix[2], np.zeros(3))
    else:
        # Transition state 0 receives the cross-link from the final stride state 2.
        assert matrix[2, 0] > 0
    model.predict(data, sampling_rate_hz=100)
    assert len(model.hidden_state_sequence_) == len(data)


def test_labeled_roth_training_supports_all_generated_transitions():
    """Combined labels may repeat a state even when each submodel learned no self-loops."""
    data = pd.DataFrame(np.random.default_rng(5).normal(size=(20, 6)), columns=BF_COLS)
    strides = pd.DataFrame({"start": [2, 7, 12, 17], "end": [4, 9, 14, 19]})
    model = RothSegmentationHmm(
        stride_model=SimpleHmm(n_states=2, n_gmm_components=1, algo_train="labeled", verbose=False),
        transition_model=SimpleHmm(
            n_states=2, n_gmm_components=1, architecture="left-right-loose", algo_train="labeled", verbose=False
        ),
        feature_transform=RothHmmFeatureTransformer(
            sampling_rate_feature_space_hz=100, low_pass_filter=None, axes=["gyr_ml"], features=["raw", "gradient"]
        ),
        algo_train="labeled",
        max_iterations=1,
        verbose=False,
    ).self_optimize([data], [strides], sampling_rate_hz=100)

    matrix = model.model.transitions
    assert np.isfinite(matrix).all()
    assert matrix[0, 0] > 0
    model.predict(data, sampling_rate_hz=100)
    assert len(model.hidden_state_sequence_) == len(data)


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
