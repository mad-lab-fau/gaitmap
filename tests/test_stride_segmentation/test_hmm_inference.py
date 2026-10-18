"""Inference results captured with pomegranate 0.14.6 before replacing its backend."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from tpcp._hash import custom_hash

from gaitmap.stride_segmentation.hmm import (
    GaussianMixture,
    HiddenMarkovModel,
    HmmStrideSegmentation,
    PreTrainedRothSegmentationModel,
    RothHmmFeatureTransformer,
    SimpleHmm,
)
from gaitmap.utils.coordinate_conversion import convert_to_fbf


def fourfold_labels(feature_labels):
    """Expand the fixture's 4:1 grid with midpoint ties assigned to the earlier label."""
    repeated = np.repeat(feature_labels, 4)
    return np.r_[repeated[1:], repeated[-1]]


def aligned_legacy_boundaries(boundaries, n_samples, old_feature_count):
    """Recover feature boundaries from the old stretched grid and map them to the complete 4:1 grid."""
    feature_boundaries = np.rint(boundaries / (n_samples / old_feature_count) + 0.5).astype(int)
    return np.clip(4 * feature_boundaries - 1, 0, n_samples)


def complete_reference_features(reference, prefix):
    """The recorded Viterbi fixtures omitted one final null-state label; MAP already kept it."""
    labels = reference[prefix + "_feature_states"]
    return np.r_[labels, labels[-1]] if prefix.endswith("viterbi") else labels


@pytest.mark.parametrize("n_states", [1, 2])
@pytest.mark.parametrize("algorithm", ["viterbi", "map"])
def test_public_prediction_keeps_every_observation_in_a_short_valid_path(n_states, algorithm):
    """A valid one-sample path survives wrappers even with more states than observations."""
    emissions = [GaussianMixture([[float(state)]], [[[1.0]]], [1.0]) for state in range(n_states)]
    model = SimpleHmm(
        n_states,
        1,
        model=HiddenMarkovModel(emissions, np.eye(n_states), np.eye(n_states)[0]),
        data_columns=("x",),
    )

    assert_array_equal(model.predict_hidden_state_sequence(pd.DataFrame({"x": [0.0]}), algorithm=algorithm), [0])


@pytest.mark.parametrize("states", [[0], [0, 1, 2], [1, 1, 0, 2, 3]])
def test_inverse_feature_transform_preserves_full_length_identity(states):
    """Decoding without downsampling needs one interpolation coordinate per label."""
    actual = RothHmmFeatureTransformer().inverse_transform_state_sequence(
        np.asarray(states), data=np.zeros(len(states))
    )

    assert_array_equal(actual, states)


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan])
def test_native_model_loader_rejects_invalid_persisted_probabilities(value):
    """Malformed saved probabilities fail at loading rather than corrupting inference."""
    emission = GaussianMixture([[0.0]], [[[1.0]]], [1.0])
    data = HiddenMarkovModel([emission], [[1.0]], [1.0]).to_dict()
    data["transitions"][0][0] = value

    with pytest.raises(ValueError, match="probabilities"):
        HiddenMarkovModel.from_dict(data)


@pytest.fixture(scope="module")
def legacy_inference():
    """Read reference results produced by the legacy backend."""
    with np.load(Path(__file__).parent / "fixtures" / "hmm_legacy_inference.npz") as reference:
        yield reference


@pytest.mark.parametrize("algorithm", ["viterbi", "map"])
@pytest.mark.parametrize("sensor", ["left_sensor", "right_sensor"])
def test_pretrained_prediction_matches_legacy(healthy_example_imu_data, legacy_inference, algorithm, sensor):
    """Preserve state labels and likelihoods for the bundled pretrained model."""
    data = convert_to_fbf(healthy_example_imu_data, left_like="left_", right_like="right_")[sensor]
    model = PreTrainedRothSegmentationModel().set_params(algo_predict=algorithm)

    result = model.predict(data, sampling_rate_hz=204.8)

    prefix = f"{sensor}_{algorithm}"
    expected_features = complete_reference_features(legacy_inference, prefix)
    assert_array_equal(result.hidden_state_sequence_feature_space_, expected_features)
    assert_array_equal(result.hidden_state_sequence_, fourfold_labels(expected_features))
    assert_allclose(
        result.model.log_probability(result.feature_space_data_.to_numpy()),
        legacy_inference[prefix + "_likelihood"],
        rtol=1e-10,
        atol=1e-8,
    )


@pytest.mark.parametrize("algorithm", ["viterbi", "map"])
def test_multisensor_segmentation_matches_legacy(healthy_example_imu_data, legacy_inference, algorithm):
    """Preserve stride boundaries for both feet and decoding algorithms."""
    data = convert_to_fbf(healthy_example_imu_data, left_like="left_", right_like="right_")
    model = PreTrainedRothSegmentationModel().set_params(algo_predict=algorithm)

    result = HmmStrideSegmentation(model, snap_to_min_win_ms=300).segment(data, sampling_rate_hz=204.8)

    for sensor in ["left_sensor", "right_sensor"]:
        prefix = f"{sensor}_{algorithm}"
        assert_array_equal(result.stride_list_[sensor].to_numpy(), legacy_inference[prefix + "_strides"])
        expected_boundaries = legacy_inference[prefix + "_original_strides"]
        if algorithm == "viterbi":
            expected_boundaries = aligned_legacy_boundaries(
                expected_boundaries, len(data[sensor]), len(legacy_inference[prefix + "_feature_states"])
            )
        assert_array_equal(result.matches_start_end_original_[sensor], expected_boundaries)
        assert_array_equal(
            result.hidden_state_sequence_[sensor],
            fourfold_labels(complete_reference_features(legacy_inference, prefix)),
        )


@pytest.mark.parametrize("restore", ["clone", "json"])
def test_restored_model_keeps_prediction_and_parameter_hash(healthy_example_imu_data, legacy_inference, restore):
    """Keep cloned and serialized models independent and prediction-compatible."""
    original = PreTrainedRothSegmentationModel()
    original_hash = custom_hash(original)
    restored = original.clone() if restore == "clone" else type(original).from_json(original.to_json())
    restored_hash = custom_hash(restored)
    data = convert_to_fbf(healthy_example_imu_data, left_like="left_", right_like="right_")["left_sensor"]

    result = restored.predict(data, sampling_rate_hz=204.8)

    assert_array_equal(
        result.hidden_state_sequence_,
        fourfold_labels(complete_reference_features(legacy_inference, "left_sensor_viterbi")),
    )
    assert custom_hash(original) == original_hash
    assert custom_hash(restored.clone()) == restored_hash
    assert restored.model is not original.model
    assert restored.stride_model.model is not original.stride_model.model
    assert restored.transition_model.model is not original.transition_model.model


def test_public_hmm_training_prediction_and_serialization_do_not_require_pomegranate():
    """Train, load and save HMMs when the old dependency is unavailable."""
    code = """
import importlib.abc
import sys

class RejectPomegranate(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "pomegranate":
            raise AssertionError("Inference attempted to import pomegranate")

sys.meta_path.insert(0, RejectPomegranate())
import numpy as np
import pandas as pd
from gaitmap.stride_segmentation.hmm import PreTrainedRothSegmentationModel, SimpleHmm
model = PreTrainedRothSegmentationModel()
model.clone()
type(model).from_json(model.to_json())
trained = SimpleHmm(2, 1, algo_train="labeled", max_iterations=1, verbose=False)
trained.self_optimize([pd.DataFrame({"feature": [-3., -1., 1., 3.]})], [np.array([0, 0, 1, 1])])
restored = SimpleHmm.from_json(trained.to_json()).clone()
np.testing.assert_array_equal(restored.predict_hidden_state_sequence(pd.DataFrame({"feature": [-3., -1., 1., 3.]})), [0, 0, 1, 1])
"""

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("algorithm", ["viterbi", "map"])
def test_saved_finite_simple_hmm_keeps_all_emitting_samples(algorithm):
    """Keep the last emitting sample when the model includes explicit termination."""
    serialized = (Path(__file__).parent / "fixtures" / "hmm_legacy_finite.json").read_text()
    model = SimpleHmm.from_json(serialized)
    data = pd.DataFrame({"feature": [-2.0, -1.0, 2.0]})

    result = model.predict_hidden_state_sequence(data, algorithm=algorithm)

    assert_array_equal(result, [0, 0, 1])
