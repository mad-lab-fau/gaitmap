"""Public training regressions against the legacy Roth training example."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from gaitmap.stride_segmentation.hmm import RothHmmFeatureTransformer, RothSegmentationHmm, SimpleHmm
from gaitmap.utils.consts import BF_COLS
from tests.test_stride_segmentation.test_hmm_inference import aligned_legacy_boundaries, fourfold_labels

matplotlib.use("Agg")


def test_simple_hmm_trains_one_multivariate_observation():
    """One observation retains its feature axis during initialization and training."""
    observations = pd.DataFrame({"x": [1.0], "y": [2.0]})
    model = SimpleHmm(1, 1, algo_train="labeled", max_iterations=1, verbose=False).self_optimize(
        [observations], [np.array([0])]
    )

    assert_array_equal(model.model.emissions[0].means, [[1.0, 2.0]])
    assert_array_equal(model.predict_hidden_state_sequence(observations), [0])


def test_transition_initialization_includes_exclusive_stride_end_samples():
    """Every observation outside half-open strides participates in transition fitting."""
    data = pd.DataFrame(np.zeros((10, 6)), columns=BF_COLS)
    data.loc[[4, 8], "gyr_ml"] = [10.0, 20.0]
    model = RothSegmentationHmm(
        stride_model=SimpleHmm(1, 1, algo_train="labeled", max_iterations=0, verbose=False),
        transition_model=SimpleHmm(
            1, 1, architecture="fully-connected", algo_train="labeled", max_iterations=0, verbose=False
        ),
        feature_transform=RothHmmFeatureTransformer(
            sampling_rate_feature_space_hz=100,
            low_pass_filter=None,
            standardization=False,
            axes=["gyr_ml"],
            features=["raw"],
        ),
        max_iterations=0,
        verbose=False,
    ).self_optimize([data], [pd.DataFrame({"start": [2, 6], "end": [4, 8]})], sampling_rate_hz=100)

    # Transition observations are [0, 0, 10, 0, 20, 0]. Stride intervals exclude their end sample.
    assert_array_equal(model.transition_model.model.emissions[0].means, [[5.0]])


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
    """Short ignored transitions can repeat the fallback label even without learned self-loops."""
    data = pd.DataFrame(np.random.default_rng(5).normal(size=(20, 6)), columns=BF_COLS)
    strides = pd.DataFrame({"start": [3, 8, 13, 17], "end": [5, 10, 15, 19]})
    model = RothSegmentationHmm(
        stride_model=SimpleHmm(n_states=2, n_gmm_components=1, algo_train="labeled", verbose=False),
        transition_model=SimpleHmm(
            n_states=3, n_gmm_components=1, architecture="left-right-loose", algo_train="labeled", verbose=False
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
    assert history.n_iterations > 0


def test_training_example_preserves_legacy_segmentation():
    """Pin measured corrections while retaining every stride in the untouched legacy reference."""
    from examples.stride_segmentation.segmentation_hmm_training import hmm

    corrections = {
        "left_sensor": {
            "classes": [*range(863, 868), *range(1772, 1777)],
            "snapped": [(13, 1, -13), (27, 1, -16)],
            "original": [(13, 1, -20), (27, 1, -20)],
        },
        "right_sensor": {
            "classes": [*range(949, 954), *range(1741, 1745), 1810],
            "snapped": [(14, 1, -14), (29, 0, 11)],
            "original": [(14, 1, -12), (15, 0, 8), (28, 1, -4), (29, 0, 12), (29, 1, 4)],
        },
    }
    with np.load(Path(__file__).parent / "fixtures" / "hmm_legacy_training_example.npz") as reference:
        for sensor in ["left_sensor", "right_sensor"]:
            # Independent small-data regressions prove complete coverage. These deltas record its real-data effect.
            feature_labels = reference[sensor + "_feature_states"]
            feature_classes = np.r_[feature_labels, feature_labels[-1]] >= 5
            feature_classes[corrections[sensor]["classes"]] ^= True
            expected_classes = fourfold_labels(feature_classes)
            assert_array_equal(hmm.hidden_state_sequence_[sensor] >= 5, expected_classes)
            expected_snapped = reference[sensor + "_strides"].copy()
            expected_original = aligned_legacy_boundaries(
                reference[sensor + "_original_strides"], len(expected_classes), len(feature_labels)
            )
            for row, column, delta in corrections[sensor]["snapped"]:
                expected_snapped[row, column] += delta
            for row, column, delta in corrections[sensor]["original"]:
                expected_original[row, column] += delta
            assert_array_equal(hmm.stride_list_[sensor].to_numpy(), expected_snapped)
            assert_array_equal(hmm.matches_start_end_original_[sensor], expected_original)
