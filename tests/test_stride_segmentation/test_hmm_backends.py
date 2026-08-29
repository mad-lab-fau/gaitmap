"""Tests for interchangeable HMM inference and training implementations."""

import hashlib
import importlib.util
import json
import sys
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest
from tpcp import clone

from gaitmap.evaluation_utils import evaluate_segmented_stride_list, precision_recall_f1_score
from gaitmap.stride_segmentation.hmm import (
    CompositeHmm,
    HmmModel,
    HmmStrideSegmentation,
    LegacyPomegranateHmmInference,
    PomegranateHmmInference,
    PomegranateHmmTrainer,
    RothHmmFeatureTransformer,
    RothSegmentationHmm,
    ScipyHmmInference,
)
from gaitmap.utils.coordinate_conversion import convert_left_foot_to_fbf, convert_right_foot_to_fbf

requires_modern_pomegranate = pytest.mark.skipif(
    sys.version_info < (3, 10) or importlib.util.find_spec("pomegranate") is None,
    reason="The torch-based pomegranate backend requires Python 3.10 or newer and the `hmm` extra.",
)
requires_legacy_pomegranate = pytest.mark.skipif(
    sys.version_info >= (3, 10) or importlib.util.find_spec("pomegranate") is None,
    reason="The legacy pomegranate backend requires Python 3.9 and the `hmm` extra.",
)
requires_python_39 = pytest.mark.skipif(
    sys.version_info >= (3, 10), reason="This compatibility boundary only exists on Python 3.9."
)


def _state_sequence_digest(state_sequence: np.ndarray) -> str:
    return hashlib.sha256(state_sequence.astype("<i8", copy=False).tobytes()).hexdigest()


def test_bundled_pretrained_model_is_the_unmodified_legacy_export() -> None:
    """The public compatibility fixture must not require an in-place format migration."""
    model_json = (
        files("gaitmap_mad.stride_segmentation.hmm._pre_trained_models")
        .joinpath("fallriskpd_at_lab_model.json")
        .read_bytes()
    )

    assert hashlib.sha256(model_json).hexdigest() == "58f06c6dcadd67a4be9256a360473869f5f3958abbcf152a4ad6e2dfcb1c7cca"
    assert RothSegmentationHmm.from_legacy_json(model_json.decode("utf8")).model is not None


def test_pretrained_scipy_inference_matches_legacy_hidden_states(healthy_example_imu_data) -> None:
    """The dependency-free decoder must reproduce the shipped legacy model exactly."""
    data = convert_left_foot_to_fbf(healthy_example_imu_data["left_sensor"])

    result = RothSegmentationHmm.from_pretrained().predict(data, sampling_rate_hz=204.8)

    assert _state_sequence_digest(result.hidden_state_sequence_feature_space_) == (
        "99f4611c46f9e10e07d93ede37eccdab8d6290b0249335c72aca22faf748f885"
    )
    assert _state_sequence_digest(result.hidden_state_sequence_) == (
        "88b2000ddf28f8444e89bfed2b953c62b3b1a557212453fff589d6a9f44a9ad9"
    )


@requires_legacy_pomegranate
def test_pretrained_legacy_backend_matches_legacy_hidden_states(healthy_example_imu_data) -> None:
    """Python 3.9 can explicitly select the old pomegranate decoder."""
    data = convert_left_foot_to_fbf(healthy_example_imu_data["left_sensor"])
    result = (
        RothSegmentationHmm.from_pretrained()
        .set_params(inference_backend=LegacyPomegranateHmmInference())
        .predict(data, sampling_rate_hz=204.8)
    )
    assert _state_sequence_digest(result.hidden_state_sequence_) == (
        "88b2000ddf28f8444e89bfed2b953c62b3b1a557212453fff589d6a9f44a9ad9"
    )


@requires_python_39
def test_modern_backends_have_an_explicit_python_39_boundary() -> None:
    """Python 3.9 users should get an actionable compatibility error."""
    with pytest.raises(RuntimeError, match=r"Python 3\.10 or newer"):
        PomegranateHmmInference().predict(RothSegmentationHmm.from_pretrained().model, pd.DataFrame())
    with pytest.raises(RuntimeError, match=r"Python 3\.10 or newer"):
        PomegranateHmmTrainer().fit(HmmModel.left_right(1, n_gmm_components=1), [], [])
    with pytest.raises(RuntimeError, match=r"Python 3\.10 or newer"):
        RothSegmentationHmm().self_optimize(
            [pd.DataFrame({"gyr_ml": np.zeros(400)})],
            [pd.DataFrame({"start": [100], "end": [300]})],
            sampling_rate_hz=204.8,
        )


def test_pretrained_scipy_inference_matches_legacy_stride_list(healthy_example_imu_data) -> None:
    """The public segmenter must preserve the legacy postprocessed stride borders."""
    data = convert_left_foot_to_fbf(healthy_example_imu_data["left_sensor"])

    result = HmmStrideSegmentation(model=RothSegmentationHmm.from_pretrained()).segment(data, sampling_rate_hz=204.8)

    assert _state_sequence_digest(result.stride_list_.to_numpy()) == (
        "14c5eafe48e0b640ebe4c841ed37f6d071b78c3dfd194cce6565447be23de526"
    )


def test_compiled_model_clone_and_json_roundtrip_preserve_inference(healthy_example_imu_data) -> None:
    """Compiled artifacts must remain portable plain-data tpcp parameters."""
    data = convert_left_foot_to_fbf(healthy_example_imu_data["left_sensor"])
    fitted_model = RothSegmentationHmm.from_pretrained().model
    assert fitted_model is not None

    for model in (clone(fitted_model), HmmModel.from_json(fitted_model.to_json())):
        result = RothSegmentationHmm(model=model).predict(data, sampling_rate_hz=204.8)
        assert _state_sequence_digest(result.hidden_state_sequence_) == (
            "88b2000ddf28f8444e89bfed2b953c62b3b1a557212453fff589d6a9f44a9ad9"
        )


@requires_modern_pomegranate
def test_pomegranate_training_compiles_to_scipy_inference_model() -> None:
    """The trainable backend must return a portable model usable by another backend."""
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(3), 30)
    samples = np.column_stack((labels * 5, labels * -3)) + rng.normal(scale=0.2, size=(len(labels), 2))
    data = pd.DataFrame(samples, columns=["feature_a", "feature_b"])

    topology = HmmModel.compose(
        parts={
            "background": HmmModel.left_right(n_states=1, n_gmm_components=1),
            "event": HmmModel.left_right(n_states=2, n_gmm_components=1),
        },
        routes={"background": ("event",)},
        starts=("background",),
        ends=("event",),
    )
    model = PomegranateHmmTrainer(max_iterations=2).fit(topology, [data], [labels])
    predicted = ScipyHmmInference().predict(model, data)

    assert model.data_columns == ("feature_a", "feature_b")
    assert model.state_ids == topology.state_ids
    assert model.state_groups == topology.state_groups
    assert np.mean(predicted == labels) > 0.95
    assert np.mean(PomegranateHmmInference().predict(model, data) == labels) > 0.95


@requires_modern_pomegranate
def test_transition_training_preserves_fitted_emissions_exactly() -> None:
    """Final composite adjustment changes probabilities but never refits submodel emissions."""
    model = HmmModel.from_parameters(
        transition_probabilities=np.array([[0.9, 0.1], [0.0, 0.8]]),
        start_probabilities=np.array([1.0, 0.0]),
        end_probabilities=np.array([0.0, 0.2]),
        means=np.array([[[0.0]], [[5.0]]]),
        covariances=np.array([[[[0.1]]], [[[0.2]]]]),
        weights=np.ones((2, 1)),
        n_gmm_components=np.ones(2, dtype=int),
        data_columns=("feature",),
    )
    labels = np.array([0, 0, 1, 1])
    data = pd.DataFrame({"feature": [0.0, 0.1, 4.9, 5.0]})

    adjusted = PomegranateHmmTrainer().fit(model, [data], [labels], train="transitions")

    np.testing.assert_array_equal(adjusted.means, model.means)
    np.testing.assert_array_equal(adjusted.covariances, model.covariances)
    np.testing.assert_array_equal(adjusted.weights, model.weights)
    np.testing.assert_allclose(adjusted.transition_probabilities, [[0.5, 0.5], [0.0, 0.5]])
    np.testing.assert_allclose(adjusted.end_probabilities, [0.0, 0.5])


@requires_modern_pomegranate
def test_composite_hmm_trains_any_number_of_named_parts_from_complete_regions() -> None:
    """Generic orchestration supports activity models without domain-specific training code."""
    part_names = ("transition", "walking", "stairs", "running")
    topology = HmmModel.compose(
        parts={name: HmmModel.left_right(1, n_gmm_components=1) for name in part_names},
        routes={
            "transition": ("walking", "stairs", "running"),
            "walking": ("transition",),
            "stairs": ("transition",),
            "running": ("transition",),
        },
        starts=("transition",),
        ends=("transition",),
    )
    region_names = ["transition", "walking", "transition", "stairs", "transition", "running", "transition"]
    regions = pd.DataFrame(
        {
            "start": np.arange(0, 140, 20),
            "end": np.arange(20, 160, 20),
            "model": region_names,
        }
    )
    means = {"transition": 0.0, "walking": 3.0, "stairs": 6.0, "running": 9.0}
    rng = np.random.default_rng(4)
    data = pd.DataFrame({"gyr_ml": np.concatenate([rng.normal(means[name], 0.1, 20) for name in region_names])})
    feature_transform = RothHmmFeatureTransformer(
        sampling_rate_feature_space_hz=20,
        low_pass_filter=None,
        axes=["gyr_ml"],
        features=["raw"],
        standardization=False,
    )

    trained = CompositeHmm(model=topology, feature_transform=feature_transform).self_optimize(
        [data], [regions], sampling_rate_hz=20
    )

    assert trained.model.is_fitted
    assert tuple(trained.model.state_groups) == part_names
    np.testing.assert_allclose(trained.model.means[:, 0, 0], [0.0, 3.0, 6.0, 9.0], atol=0.1)


@pytest.mark.parametrize(
    ("regions", "error"),
    [
        (pd.DataFrame({"start": [0, 3], "end": [2, 4], "model": ["a", "b"]}), "no gaps"),
        (pd.DataFrame({"start": [0, 1], "end": [2, 4], "model": ["a", "b"]}), "non-overlapping"),
        (pd.DataFrame({"start": [0], "end": [4], "model": ["unknown"]}), "unknown model parts"),
        (pd.DataFrame({"start": [0], "end": [4]}), "missing required columns"),
        (pd.DataFrame({"start": [0.0], "end": [3.5], "model": ["a"]}), "integer sample indices"),
    ],
)
def test_composite_hmm_rejects_invalid_region_contracts(regions, error) -> None:
    """Generic training never guesses how gaps, overlaps, or unknown semantic labels should be handled."""
    topology = HmmModel.compose(
        parts={"a": HmmModel.left_right(1, n_gmm_components=1), "b": HmmModel.left_right(1, n_gmm_components=1)},
        routes={"a": ("b",)},
        starts=("a",),
        ends=("b",),
    )

    with pytest.raises(ValueError, match=error):
        CompositeHmm(model=topology).self_optimize(
            [pd.DataFrame({"gyr_ml": np.arange(4)})], [regions], sampling_rate_hz=204.8
        )


def test_composite_orchestration_rejects_parts_without_an_ordered_initialization_path() -> None:
    """Arbitrary graphs require caller-supplied labels through the atomic trainer."""
    reverse_part = HmmModel.from_matrix(
        allowed_transitions=np.array([[1, 0], [1, 1]], dtype=bool),
        allowed_starts=np.array([0, 1], dtype=bool),
        allowed_ends=np.array([1, 0], dtype=bool),
        n_gmm_components=np.ones(2),
    )
    topology = HmmModel.compose(parts={"reverse": reverse_part}, routes={}, starts=("reverse",), ends=("reverse",))

    with pytest.raises(ValueError, match="ordered initialization"):
        CompositeHmm(model=topology).self_optimize(
            [pd.DataFrame({"gyr_ml": np.arange(4)})],
            [pd.DataFrame({"start": [0], "end": [4], "model": ["reverse"]})],
            sampling_rate_hz=204.8,
        )


def test_stride_segmentation_emits_types_for_multiple_selected_state_groups() -> None:
    """One generic segmenter can expose walking and running bouts without a domain-specific subclass."""
    fitted_model = HmmModel.from_parameters(
        transition_probabilities=np.array([[0.8, 0.1, 0.1], [0.2, 0.8, 0.0], [0.2, 0.0, 0.8]]),
        start_probabilities=np.array([1.0, 0.0, 0.0]),
        end_probabilities=np.zeros(3),
        means=np.array([[[0.0]], [[5.0]], [[10.0]]]),
        covariances=np.array([[[[0.1]]], [[[0.1]]], [[[0.1]]]]),
        weights=np.ones((3, 1)),
        n_gmm_components=np.ones(3, dtype=int),
        data_columns=("raw__gyr_ml",),
        state_groups={"walking": (1,), "running": (2,)},
    )
    feature_transform = RothHmmFeatureTransformer(
        sampling_rate_feature_space_hz=20,
        low_pass_filter=None,
        axes=["gyr_ml"],
        features=["raw"],
        standardization=False,
    )
    data = pd.DataFrame({"gyr_ml": np.repeat([0.0, 5.0, 0.0, 10.0, 0.0], 5)})
    model = CompositeHmm(model=fitted_model, feature_transform=feature_transform)

    result = HmmStrideSegmentation(
        model=model, segment_state_groups=("walking", "running"), snap_to_min_win_ms=None
    ).segment(data, sampling_rate_hz=20)

    assert tuple(result.stride_list_.columns) == ("start", "end", "type")
    assert result.stride_list_["type"].tolist() == ["walking", "running"]


def test_segmentation_uses_contiguous_membership_in_the_complete_state_group() -> None:
    """Segments may enter or leave through any state and split whenever group membership ends."""
    fitted_model = HmmModel.from_parameters(
        transition_probabilities=np.full((4, 4), 0.25),
        start_probabilities=np.full(4, 0.25),
        end_probabilities=np.zeros(4),
        means=np.arange(4, dtype=float)[:, None, None],
        covariances=np.repeat(np.array([[[[0.01]]]]), 4, axis=0),
        weights=np.ones((4, 1)),
        n_gmm_components=np.ones(4, dtype=int),
        data_columns=("raw__gyr_ml",),
        state_groups={"activity": (1, 2)},
    )
    feature_transform = RothHmmFeatureTransformer(
        sampling_rate_feature_space_hz=20,
        low_pass_filter=None,
        axes=["gyr_ml"],
        features=["raw"],
        standardization=False,
    )
    data = pd.DataFrame({"gyr_ml": np.repeat([0.0, 2.0, 0.0, 1.0, 3.0, 2.0], 5)})

    result = HmmStrideSegmentation(
        model=CompositeHmm(model=fitted_model, feature_transform=feature_transform),
        segment_state_groups=("activity",),
        segment_group_mode="membership",
        snap_to_min_win_ms=None,
    ).segment(data, sampling_rate_hz=20)

    np.testing.assert_array_equal(result.stride_list_.to_numpy(), [[5, 10], [16, 21], [26, 30]])


def test_roth_default_topology_accepts_strides_at_recording_boundaries() -> None:
    """Stride annotations may start at zero or end at the final sample."""
    model = RothSegmentationHmm().model

    assert model.allowed_starts[model.state_indices("stride")[0]]
    assert model.allowed_ends[model.state_indices("stride")[-1]]


@requires_modern_pomegranate
def test_roth_model_trains_and_segments_with_modern_pomegranate(
    healthy_example_imu_data, healthy_example_stride_borders, monkeypatch
) -> None:
    """Roth's public optimization path must train a model consumable by segmentation."""
    data = convert_left_foot_to_fbf(healthy_example_imu_data["left_sensor"])
    training = PomegranateHmmTrainer(max_iterations=1)
    fit_calls = []
    fit = PomegranateHmmTrainer.fit

    def record_fit(self, *args, **kwargs):
        fit_calls.append(kwargs.get("train", "all"))
        return fit(self, *args, **kwargs)

    monkeypatch.setattr(PomegranateHmmTrainer, "fit", record_fit)
    model = RothSegmentationHmm(training_backend=training).self_optimize(
        [data], [healthy_example_stride_borders["left_sensor"]], sampling_rate_hz=204.8
    )

    test_data = convert_right_foot_to_fbf(healthy_example_imu_data["right_sensor"])
    result = HmmStrideSegmentation(model=model).segment(test_data, sampling_rate_hz=204.8)
    ground_truth = healthy_example_stride_borders["right_sensor"].set_index("s_id")
    matches = evaluate_segmented_stride_list(
        segmented_stride_list=result.stride_list_, ground_truth=ground_truth, tolerance=0.2
    )

    assert model.model is not None
    assert model.model.stride_states == tuple(range(5, 25))
    assert fit_calls == ["all", "all", "transitions"]
    assert precision_recall_f1_score(matches)["f1_score"] > 0.9


@requires_modern_pomegranate
@pytest.mark.parametrize("labels", [np.array([], dtype=int), np.array([[0, 1]])])
def test_pomegranate_training_rejects_invalid_label_shapes(labels) -> None:
    """Training validation should fail before entering pomegranate internals."""
    data = pd.DataFrame(np.empty((labels.size, 1)), columns=["feature"])
    with pytest.raises(ValueError, match=r"non-empty.*one-dimensional"):
        PomegranateHmmTrainer().fit(HmmModel.left_right(1, n_gmm_components=1), [data], [labels])


@requires_modern_pomegranate
def test_legacy_json_loads_into_every_inference_backend() -> None:
    """Old pomegranate JSON must migrate once into the shared compiled model."""

    def distribution(mean):
        return {
            "class": "Distribution",
            "name": "MultivariateGaussianDistribution",
            "parameters": [[mean], [[0.1]]],
            "frozen": False,
        }

    legacy_json = json.dumps(
        {
            "_gaitmap_obj": "RothSegmentationHmm",
            "params": {
                "data_columns": ["feature"],
                "feature_transform": {
                    "_gaitmap_obj": "RothHmmFeatureTransformer",
                    "params": {"axes": ["gyr_ml"], "features": ["raw"], "standardization": True},
                },
                "transition_model": {"params": {"n_states": 1}},
                "stride_model": {"params": {"n_states": 1}},
                "model": {
                    "_obj_type": "HiddenMarkovModel",
                    "hmm": {
                        "states": [
                            {"name": "s0", "distribution": distribution(0)},
                            {"name": "s1", "distribution": distribution(5)},
                            {"name": "None-start", "distribution": None},
                            {"name": "None-end", "distribution": None},
                        ],
                        "start_index": 2,
                        "end_index": 3,
                        "edges": [
                            [2, 0, 1.0, 1.0, None],
                            [0, 0, 0.8, 1.0, None],
                            [0, 1, 0.2, 1.0, None],
                            [1, 1, 1.0, 1.0, None],
                        ],
                    },
                },
            },
        }
    )
    loaded = RothSegmentationHmm.from_legacy_json(legacy_json)
    data = pd.DataFrame({"feature": [0.0] * 10 + [5.0] * 10})

    for backend in (ScipyHmmInference(), PomegranateHmmInference()):
        states = backend.predict(loaded.model, data)
        assert set(states) <= {0, 1}
        assert len(states) >= len(data) - 1
