"""Tests for the optional native pomegranate 1.x inference path."""

import numpy as np
import pytest

pytest.importorskip("gaitmap_mad.stride_segmentation.hmm.modern")
import torch
from gaitmap_mad.stride_segmentation.hmm import PreTrainedRothSegmentationModel
from gaitmap_mad.stride_segmentation.hmm._backend_common import prepare_predict_data
from gaitmap_mad.stride_segmentation.hmm._state import FlatHmmState, GaussianEmissionState, HmmGraphState
from gaitmap_mad.stride_segmentation.hmm.modern import PomegranateModernHmmBackend
from gaitmap_mad.stride_segmentation.hmm.modern._state import flat_hmm_state_to_pomegranate_modern_model
from gaitmap_mad.stride_segmentation.hmm.scipy._utils import log_emission_probabilities

from gaitmap.example_data import get_healthy_example_imu_data
from gaitmap.utils.coordinate_conversion import convert_left_foot_to_fbf


def _viterbi_decode_with_end_probs(model, log_emissions: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        transition_log_probs = np.log(np.asarray(model.compiled.graph.transition_probs, dtype=float))
        start_log_probs = np.log(np.asarray(model.compiled.graph.start_probs, dtype=float))
        end_log_probs = np.log(np.asarray(model.compiled.graph.end_probs, dtype=float))

    n_samples, n_states = log_emissions.shape
    dp = np.full((n_samples, n_states), -np.inf, dtype=float)
    pointers = np.zeros((n_samples, n_states), dtype=int)

    dp[0] = start_log_probs + log_emissions[0]
    for sample_idx in range(1, n_samples):
        scores = dp[sample_idx - 1][:, None] + transition_log_probs
        pointers[sample_idx] = np.argmax(scores, axis=0)
        dp[sample_idx] = scores[pointers[sample_idx], np.arange(n_states)] + log_emissions[sample_idx]

    path = np.zeros(n_samples, dtype=int)
    path[-1] = int(np.argmax(dp[-1] + end_log_probs))
    for sample_idx in range(n_samples - 1, 0, -1):
        path[sample_idx - 1] = pointers[sample_idx, path[sample_idx]]
    return path


def _get_pretrained_feature_data():
    model = PreTrainedRothSegmentationModel()
    data = convert_left_foot_to_fbf(get_healthy_example_imu_data()["left_sensor"])
    feature_data, _ = model._transform([data], None, sampling_rate_hz=100)
    return model, feature_data[0]


def test_modern_native_map_matches_canonical_backend() -> None:
    """The native MAP path should match the canonical decoder exactly."""
    model, feature_data = _get_pretrained_feature_data()

    canonical = PomegranateModernHmmBackend().predict(
        model.model,
        feature_data,
        expected_columns=model.data_columns,
        algorithm="map",
        verbose=model.verbose,
    )
    native = PomegranateModernHmmBackend(inference_implementation="native").predict(
        model.model,
        feature_data,
        expected_columns=model.data_columns,
        algorithm="map",
        verbose=model.verbose,
    )

    np.testing.assert_array_equal(canonical, native)


def test_modern_native_viterbi_matches_full_length_end_probability_decode() -> None:
    """The native Viterbi path should use the full-length end-probability decode."""
    model, feature_data = _get_pretrained_feature_data()

    canonical = PomegranateModernHmmBackend().predict(
        model.model,
        feature_data,
        expected_columns=model.data_columns,
        algorithm="viterbi",
        verbose=model.verbose,
    )
    native = PomegranateModernHmmBackend(inference_implementation="native").predict(
        model.model,
        feature_data,
        expected_columns=model.data_columns,
        algorithm="viterbi",
        verbose=model.verbose,
    )

    observations = prepare_predict_data(feature_data, model.data_columns, len(model.model.compiled.state_names))
    log_emissions = log_emission_probabilities(model.model, observations)
    expected_native = _viterbi_decode_with_end_probs(model.model, log_emissions)

    assert len(native) == len(canonical)
    np.testing.assert_array_equal(native, expected_native)
    np.testing.assert_array_equal(canonical, expected_native)


def test_modern_runtime_model_uses_float64_for_training() -> None:
    """Modern runtime models should train without dtype mismatches on float64 data."""
    state = FlatHmmState(
        graph=HmmGraphState(
            transition_probs=np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64),
            start_probs=np.array([1.0, 0.0], dtype=np.float64),
            end_probs=np.array([0.0, 1.0], dtype=np.float64),
        ),
        emissions=(
            GaussianEmissionState(
                mean=np.array([0.0], dtype=np.float64),
                covariance=np.array([[1.0]], dtype=np.float64),
            ),
            GaussianEmissionState(
                mean=np.array([1.0], dtype=np.float64),
                covariance=np.array([[1.0]], dtype=np.float64),
            ),
        ),
        state_names=("s0", "s1"),
        name="dtype_regression",
    )
    runtime_model = flat_hmm_state_to_pomegranate_modern_model(state)
    training_data = np.array([[[0.1], [0.2], [1.1]]], dtype=np.float64)
    priors = np.array([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=np.float64)

    assert runtime_model.dtype == torch.float64

    runtime_model.fit(training_data, priors=priors)
