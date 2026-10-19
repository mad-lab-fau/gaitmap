"""Scientific regressions through gaitmap HMM training interfaces."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from gaitmap.stride_segmentation.hmm import (
    RothHmmFeatureTransformer,
    RothSegmentationHmm,
    SimpleHmm,
)
from gaitmap.utils.consts import BF_COLS


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

    for observations, expected in zip(data, labels, strict=True):
        assert_array_equal(trained.predict_hidden_state_sequence(observations), expected)
        assert_array_equal(restored.predict_hidden_state_sequence(observations), expected)
    assert history.n_iterations > 0


@pytest.mark.parametrize("algorithm", ["viterbi", "map"])
@pytest.mark.parametrize("short_region", ["transition", "stride"])
def test_roth_training_uses_valid_single_sample_regions(algorithm, short_region):
    """A short region's predicted state determines the combined model's cross-link."""
    if short_region == "transition":
        values = [0.0, 0.0, 10.0, 10.0, 100.0, 100.0, 10.0, 100.0, 100.0, 0.0, 0.0, 10.0, 10.0]
        strides = pd.DataFrame({"start": [4, 7], "end": [6, 9]})
        n_stride_states, n_transition_states = 1, 2
        # The gap at index 6 belongs to transition state 1, following stride state 2.
        expected_edge = (2, 1)
    else:
        values = [
            -100.0,
            -100.0,
            0.0,
            0.0,
            100.0,
            100.0,
            -100.0,
            -100.0,
            100.0,
            -100.0,
            -100.0,
            0.0,
            0.0,
            100.0,
            100.0,
            -100.0,
            -100.0,
        ]
        strides = pd.DataFrame({"start": [2, 8, 11], "end": [6, 9, 15]})
        n_stride_states, n_transition_states = 2, 1
        # The stride at index 8 belongs to stride state 2, following transition state 0.
        expected_edge = (0, 2)
    data = pd.DataFrame(np.zeros((len(values), 6)), columns=BF_COLS)
    data["gyr_ml"] = values
    model = RothSegmentationHmm(
        stride_model=SimpleHmm(n_stride_states, 1, architecture="fully-connected", max_iterations=0, verbose=False),
        transition_model=SimpleHmm(
            n_transition_states, 1, architecture="fully-connected", max_iterations=0, verbose=False
        ),
        feature_transform=RothHmmFeatureTransformer(
            sampling_rate_feature_space_hz=100,
            low_pass_filter=None,
            standardization=False,
            axes=["gyr_ml"],
            features=["raw"],
        ),
        algo_predict=algorithm,
        max_iterations=0,
        verbose=False,
    ).self_optimize([data], [strides], sampling_rate_hz=100)

    assert model.model.transitions[expected_edge] > 0


@pytest.mark.parametrize("algorithm", ["baum-welch", "viterbi", "labeled"])
def test_mixture_initialization_and_training_are_repeatable_across_worker_counts(algorithm):
    """Repeated fits preserve densities and state assignments, including singular features."""
    rng = np.random.default_rng(16)
    labels = [np.repeat([0, 1], repeats) for repeats in [20, 30]]
    values = [rng.normal(0, 0.2, len(path)) + 10 * path for path in labels]
    observations = [pd.DataFrame({"x": x, "y": 2 * x}) for x in values]
    template = SimpleHmm(n_states=2, n_gmm_components=2, algo_train=algorithm, max_iterations=2, verbose=False)

    serial, serial_history = template.clone().self_optimize_with_info(observations, labels)
    parallel, parallel_history = template.clone().set_params(n_jobs=2).self_optimize_with_info(observations, labels)

    assert_array_equal(parallel.model.transitions, serial.model.transitions)
    assert_array_equal(parallel.model.starts, serial.model.starts)
    assert_array_equal(parallel.model.ends, serial.model.ends)
    assert_array_equal(parallel_history.log_likelihoods, serial_history.log_likelihoods)
    for actual, expected in zip(parallel.model.emissions, serial.model.emissions, strict=True):
        assert_array_equal(actual.means, expected.means)
        assert_array_equal(actual.covariances, expected.covariances)
        assert_array_equal(actual.weights, expected.weights)
    for data, expected in zip(observations, labels, strict=True):
        assert_array_equal(serial.predict_hidden_state_sequence(data), expected)
        assert_array_equal(parallel.predict_hidden_state_sequence(data), expected)
