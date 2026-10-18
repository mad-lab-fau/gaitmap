"""Public training regressions against the legacy Roth training example."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from gaitmap_mad.stride_segmentation.hmm._hmm_distributions import GaussianMixture
from gaitmap_mad.stride_segmentation.hmm._hmm_inference import HiddenMarkovModel
from gaitmap_mad.stride_segmentation.hmm._segmentation_model import create_fully_labeled_gait_sequences
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


def region_model(means, *, strict=False, unreachable_end=False):
    """Build small trained region models with independently known Gaussian state labels."""
    emissions = [GaussianMixture([[mean]], [[[1.0]]], [1.0]) for mean in means]
    n_states = len(means)
    if strict:
        transitions = [[1.0, 0.0], [0.0, 0.5]] if unreachable_end else [[0.5, 0.5], [0.0, 0.5]]
        starts = [1.0, 0.0]
        ends = [0.0, 0.5]
    else:
        transitions = np.full((n_states, n_states), 0.5 / n_states)
        starts = np.full(n_states, 1.0 / n_states)
        ends = np.full(n_states, 0.5)
    model = HiddenMarkovModel(emissions, transitions, starts, ends)
    return SimpleHmm(n_states, 1, model=model, data_columns=("x",), verbose=False)


@pytest.mark.parametrize("algorithm", ["viterbi", "map"])
@pytest.mark.parametrize("short_region", ["transition", "stride"])
def test_combined_training_keeps_valid_short_region_predictions(algorithm, short_region):
    """Decode partial regions when the learned topology admits a shorter valid path."""
    if short_region == "transition":
        transition = region_model([-100.0, 10.0])
        stride = region_model([100.0])
        data = pd.DataFrame({"x": [10.0, 100.0, 100.0, 10.0]})
        strides = pd.DataFrame({"start": [1], "end": [3]})
        expected = [1, 2, 2, 1]
    else:
        transition = region_model([10.0])
        stride = region_model([-100.0, 100.0])
        data = pd.DataFrame({"x": [10.0, 100.0, 10.0]})
        strides = pd.DataFrame({"start": [1], "end": [2]})
        expected = [0, 2, 0]

    labels = create_fully_labeled_gait_sequences([data], [strides], transition, stride, algorithm)

    assert_array_equal(labels[0], expected)


def test_combined_training_preserves_fallback_for_impossible_short_paths():
    """Leave genuinely undecodable short regions at the existing transition fallback label."""
    transition = region_model([-100.0, 10.0], strict=True)
    stride = region_model([100.0])
    data = pd.DataFrame({"x": [10.0, 100.0, 100.0, 10.0]})
    strides = pd.DataFrame({"start": [1], "end": [3]})

    labels = create_fully_labeled_gait_sequences([data], [strides], transition, stride, "viterbi")

    assert_array_equal(labels[0], [0, 2, 2, 0])


def test_combined_training_reports_impossible_paths_in_longer_regions():
    """Do not hide an unreachable learned topology when region length is sufficient."""
    transition = region_model([-100.0, 10.0], strict=True, unreachable_end=True)
    stride = region_model([100.0])
    data = pd.DataFrame({"x": [10.0, 10.0, 100.0, 100.0]})
    strides = pd.DataFrame({"start": [2], "end": [4]})

    with pytest.raises(ValueError, match="No valid state path"):
        create_fully_labeled_gait_sequences([data], [strides], transition, stride, "viterbi")


def test_combined_training_does_not_ignore_invalid_observations_in_short_regions():
    """Observation validation errors propagate instead of being treated as impossible topology."""
    transition = region_model([-100.0, 10.0])
    stride = region_model([100.0])
    data = pd.DataFrame({"x": [np.nan, 100.0, 100.0, 10.0]})
    strides = pd.DataFrame({"start": [1], "end": [3]})

    with pytest.raises(ValueError, match="finite"):
        create_fully_labeled_gait_sequences([data], [strides], transition, stride, "viterbi")


def test_roth_training_uses_valid_single_sample_transition_state():
    """Build the combined topology from a valid short gap's actual nonzero state label."""
    values = [0.0, 0.0, 10.0, 10.0, 100.0, 100.0, 10.0, 100.0, 100.0, 0.0, 0.0, 10.0, 10.0]
    data = pd.DataFrame(np.zeros((len(values), 6)), columns=BF_COLS)
    data["gyr_ml"] = values
    model = RothSegmentationHmm(
        stride_model=SimpleHmm(1, 1, max_iterations=0, verbose=False),
        transition_model=SimpleHmm(2, 1, architecture="fully-connected", max_iterations=0, verbose=False),
        feature_transform=RothHmmFeatureTransformer(
            sampling_rate_feature_space_hz=100,
            low_pass_filter=None,
            standardization=False,
            axes=["gyr_ml"],
            features=["raw"],
        ),
        max_iterations=0,
        verbose=False,
    ).self_optimize([data], [pd.DataFrame({"start": [4, 7], "end": [6, 9]})], sampling_rate_hz=100)

    # The single-sample gap at index 6 belongs to transition state 1.
    # It therefore creates a stride-state 2 -> transition-state 1 cross-link.
    assert model.model.transitions[2, 1] > 0
