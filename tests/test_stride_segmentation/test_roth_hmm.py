import json
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from gaitmap_mad.stride_segmentation.hmm import (
    CompositeHmmConfig,
    HMMState,
    HmmStrideSegmentation,
    HmmSubModelConfig,
    PreTrainedRothSegmentationModel,
    RothHmmConfig,
    RothHmmFeatureTransformer,
    RothSegmentationHmm,
    get_default_hmm_backend,
)
from gaitmap_mad.stride_segmentation.hmm.scipy import ScipyHmmInferenceBackend
from numpy.testing import assert_almost_equal, assert_array_equal
from pandas.testing import assert_frame_equal
from tpcp._hash import custom_hash

from gaitmap.data_transform import SlidingWindowMean
from gaitmap.utils.consts import BF_COLS
from gaitmap.utils.coordinate_conversion import convert_left_foot_to_fbf, convert_to_fbf
from gaitmap.utils.datatype_helper import (
    get_multi_sensor_names,
    is_multi_sensor_stride_list,
    is_single_sensor_stride_list,
)
from gaitmap.utils.exceptions import ValidationError
from tests._hmm_test_helpers import (
    import_legacy_hmm_backend,
    load_pretrained_inference_stride_list_snapshot,
)
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin

# Fix random seed for reproducibility
np.random.seed(1)

try:
    PomegranateLegacyHmmBackend = import_module(
        "gaitmap_mad.stride_segmentation.hmm.legacy"
    ).PomegranateLegacyHmmBackend
except ImportError:
    PomegranateLegacyHmmBackend = None

try:
    PomegranateModernHmmBackend = import_module(
        "gaitmap_mad.stride_segmentation.hmm.modern"
    ).PomegranateModernHmmBackend
except ImportError:
    PomegranateModernHmmBackend = None


def _require_legacy_backend():
    import_legacy_hmm_backend()
    backend_module = import_module("gaitmap_mad.stride_segmentation.hmm.legacy._backend")
    return SimpleNamespace(
        backend_class=backend_module.PomegranateLegacyHmmBackend,
        initialize_hmm=backend_module.initialize_hmm,
    )


def _default_backend_type():
    return type(get_default_hmm_backend())


def _runtime_inference_backend_params():
    params = [pytest.param(ScipyHmmInferenceBackend(), id="scipy")]
    if PomegranateModernHmmBackend is not None:
        params.append(pytest.param(PomegranateModernHmmBackend(), id="pomegranate-modern"))
    return params


def _trainable_backend_params():
    params = []
    if PomegranateLegacyHmmBackend is not None:
        params.append(pytest.param(PomegranateLegacyHmmBackend(), id="pomegranate-legacy"))
    if PomegranateModernHmmBackend is not None:
        params.append(pytest.param(PomegranateModernHmmBackend(), id="pomegranate-modern"))
    if not params:
        params.append(
            pytest.param(
                None, id="no-trainable-backend", marks=pytest.mark.skip(reason="requires a trainable HMM backend")
            )
        )
    return params


def _create_roth_model_config(*, stride_n_states=20, stride_n_gmm_components=6, transition_n_gmm_components=3):
    return CompositeHmmConfig(
        modules=(
            HmmSubModelConfig(
                name="transition",
                role="transition",
                n_states=5,
                n_gmm_components=transition_n_gmm_components,
                algo_train="baum-welch",
                stop_threshold=1e-9,
                max_iterations=10,
                architecture="left-right-loose",
            ),
            HmmSubModelConfig(
                name="stride",
                role="stride",
                n_states=stride_n_states,
                n_gmm_components=stride_n_gmm_components,
                algo_train="baum-welch",
                stop_threshold=1e-9,
                max_iterations=10,
                architecture="left-right-strict",
            ),
        )
    )


def _create_roth_hmm_config(*, stride_n_states=20, stride_n_gmm_components=6, transition_n_gmm_components=3):
    return RothHmmConfig(
        model_config=_create_roth_model_config(
            stride_n_states=stride_n_states,
            stride_n_gmm_components=stride_n_gmm_components,
            transition_n_gmm_components=transition_n_gmm_components,
        )
    )


def _stride_list_to_region_list(stride_list: pd.DataFrame, region_type: str = "stride") -> pd.DataFrame:
    region_list = stride_list[["start", "end"]].copy()
    region_list.insert(0, "roi_id", np.arange(len(region_list)))
    region_list["type"] = region_type
    return region_list.set_index("roi_id")


def _new_trainable_hmm(
    backend,
    *,
    n_states: int = 5,
    n_gmm_components: int = 3,
    architecture: str = "left-right-strict",
    algo_train: str = "baum-welch",
    stop_threshold: float = 1e-9,
    max_iterations: int = int(1e8),
    verbose: bool = True,
    n_jobs: int = 1,
    name: str = "my_model",
):
    return backend.create_submodel(
        HmmSubModelConfig(
            name=name,
            role="stride",
            n_states=n_states,
            n_gmm_components=n_gmm_components,
            architecture=architecture,
            algo_train=algo_train,
            stop_threshold=stop_threshold,
            max_iterations=max_iterations,
            verbose=verbose,
            n_jobs=n_jobs,
        )
    )


class TestMetaFunctionalityRothSegmentationHmm(TestAlgorithmMixin):
    __test__ = True

    algorithm_class = RothSegmentationHmm

    @pytest.fixture
    def after_action_instance(self, healthy_example_imu_data) -> RothSegmentationHmm:
        hmm = PreTrainedRothSegmentationModel()
        hmm.predict(convert_left_foot_to_fbf(healthy_example_imu_data["left_sensor"]), sampling_rate_hz=100)
        return hmm


class TestMetaFunctionalityHmmStrideSegmentation(TestAlgorithmMixin):
    __test__ = True

    algorithm_class = HmmStrideSegmentation

    @pytest.fixture
    def after_action_instance(self, healthy_example_imu_data) -> HmmStrideSegmentation:
        hmm = HmmStrideSegmentation()
        hmm.segment(convert_left_foot_to_fbf(healthy_example_imu_data["left_sensor"]), sampling_rate_hz=100)
        return hmm


class TestMetaFunctionalityRothHMMFeatureTransformer(TestAlgorithmMixin):
    __test__ = True

    algorithm_class = RothHmmFeatureTransformer

    @pytest.fixture
    def after_action_instance(
        self, healthy_example_imu_data, healthy_example_stride_borders
    ) -> RothHmmFeatureTransformer:
        transform = RothHmmFeatureTransformer()
        transform.transform(
            convert_left_foot_to_fbf(healthy_example_imu_data["left_sensor"]),
            roi_list=healthy_example_stride_borders["left_sensor"],
            sampling_rate_hz=100,
        )
        return transform


class TestRothHmmFeatureTransform:
    @pytest.mark.parametrize("target_sampling_rate", [50, 25, 16.3])
    def test_inverse_transform_state_sequence(self, target_sampling_rate) -> None:
        transform = RothHmmFeatureTransformer(sampling_rate_feature_space_hz=target_sampling_rate)
        in_state_sequence = np.array([0, 1, 2, 2, 4, 5])
        len_data = int(len(in_state_sequence) * np.round(100 / target_sampling_rate))
        data = np.zeros(len_data)
        state_sequence = transform.inverse_transform_state_sequence(state_sequence=in_state_sequence, data=data)
        # output should not contain any other value than in_state_sequence
        assert_array_equal(np.unique(state_sequence), np.unique(in_state_sequence))
        # output should have the same length as original data
        assert len(state_sequence) == len(data)

    @pytest.mark.parametrize("features", [["raw"], ["raw", "gradient"], ["raw", "gradient", "mean"]])
    @pytest.mark.parametrize("axes", [["gyr_ml"], ["acc_pa"], ["gyr_ml", "acc_pa"]])
    def test_select_features(self, features, healthy_example_imu_data, axes) -> None:
        transform = RothHmmFeatureTransformer(
            features=features,
            axes=axes,
        )
        data = convert_left_foot_to_fbf(healthy_example_imu_data["left_sensor"]).iloc[:100]

        transformed_data = transform.transform(data, sampling_rate_hz=100).transformed_data_

        feature_prefixes = {"raw": "", "gradient": "__gradient", "mean": "__mean"}

        assert transformed_data.shape[1] == len(features) * len(axes) == transform.n_features
        assert set(transformed_data.columns) == {
            f"{feature}{feature_prefixes[feature]}__{axis}" for feature in features for axis in axes
        }

    def test_actual_output(self, healthy_example_imu_data) -> None:
        # We disable downsampling, standardization, and filtering for this test
        transform = RothHmmFeatureTransformer(
            sampling_rate_feature_space_hz=100,
            standardization=False,
            low_pass_filter=None,
            features=["raw", "mean"],
            axes=["gyr_ml", "acc_pa"],
        )
        data = convert_left_foot_to_fbf(healthy_example_imu_data["left_sensor"]).iloc[:100]

        transformed_data = transform.transform(data, sampling_rate_hz=100).transformed_data_

        assert_array_equal(transformed_data["raw__gyr_ml"], data["gyr_ml"])
        assert_array_equal(transformed_data["raw__acc_pa"], data["acc_pa"])
        assert_array_equal(
            transformed_data["mean__mean__gyr_ml"],
            SlidingWindowMean(window_size_s=transform.window_size_s)
            .transform(data["gyr_ml"], sampling_rate_hz=100)
            .transformed_data_,
        )
        assert_array_equal(
            transformed_data["mean__mean__acc_pa"],
            SlidingWindowMean(window_size_s=transform.window_size_s)
            .transform(data["acc_pa"], sampling_rate_hz=100)
            .transformed_data_,
        )

        assert transform.data is data

    def test_type_error_filter(self) -> None:
        with pytest.raises(TypeError) as e:
            RothHmmFeatureTransformer(low_pass_filter="test").transform([], sampling_rate_hz=100)

        assert "low_pass_filter" in str(e.value)

    @pytest.mark.parametrize(("roi", "data"), [(None, []), ([], None)])
    def test_value_error_missing_sampling_rate(self, roi, data) -> None:
        with pytest.raises(ValueError) as e:
            RothHmmFeatureTransformer().transform(data, roi_list=roi, sampling_rate_hz=None)

        assert "sampling_rate_hz" in str(e.value)

    def test_resample_roi(self) -> None:
        transform = RothHmmFeatureTransformer(sampling_rate_feature_space_hz=50)
        roi = pd.DataFrame(np.array([[0, 100], [200, 300], [400, 500]]), columns=["start", "end"])
        resampled_roi = transform.transform(roi_list=roi, sampling_rate_hz=100).transformed_roi_list_
        assert_array_equal(resampled_roi, np.array([[0, 50], [100, 150], [200, 250]]))
        assert transform.roi_list is roi
        assert transform.sampling_rate_hz == 100


class TestTrainableHmmBackends:
    @pytest.mark.parametrize("backend", _trainable_backend_params())
    def test_error_on_different_number_data_and_labels(self, backend) -> None:
        with pytest.raises(ValueError) as e:
            _new_trainable_hmm(backend).self_optimize(
                [np.random.rand(100, 3)], [np.random.rand(100), np.random.rand(100)]
            )

        assert "The given training sequence and initial training labels" in str(e.value)

    @pytest.mark.parametrize(
        "data",
        [
            pd.DataFrame(np.random.rand(100, 3), columns=["feature1", "feature2", "feature3"]),
            pd.DataFrame(np.random.rand(100, 1), columns=["feature1"]),
        ],
    )
    @pytest.mark.parametrize("n_gmm_components", [1, 3])
    @pytest.mark.parametrize("n_states", [5, 12])
    @pytest.mark.parametrize("backend", _trainable_backend_params())
    def test_optimize_with_single_sequence(self, data, n_gmm_components, n_states, backend) -> None:
        model = _new_trainable_hmm(
            backend,
            n_states=n_states,
            n_gmm_components=n_gmm_components,
            max_iterations=1,
        )
        model.self_optimize([data], [pd.Series(np.tile(np.arange(n_states), int(np.ceil(100 / n_states)))[:100])])

        assert list(model.data_columns) == data.columns.tolist()
        prediction = model.predict_hidden_state_sequence(data)
        assert len(prediction) == len(data)
        assert set(prediction).issubset(set(range(n_states)))

    @pytest.mark.parametrize("backend", _trainable_backend_params())
    def test_predict_raises_error_without_optimize(self, backend) -> None:
        with pytest.raises(ValueError) as e:
            _new_trainable_hmm(backend).predict_hidden_state_sequence(pd.DataFrame(np.random.rand(100, 3)))

        assert "Call `self_optimize` first" in str(e.value) or "You need to train the HMM" in str(e.value)

    @pytest.mark.parametrize("backend", _trainable_backend_params())
    def test_predict_raises_error_on_invalid_columns(self, backend) -> None:
        model = _new_trainable_hmm(backend)
        col_names = ["feature1", "feature2", "feature3"]
        invalid_col_names = ["feature1", "feature2", "feature4"]
        model.self_optimize(
            [pd.DataFrame(np.random.rand(100, 3), columns=col_names)], [pd.Series(np.random.choice(5, 100))]
        )
        with pytest.raises(ValueError) as e:
            model.predict_hidden_state_sequence(pd.DataFrame(np.random.rand(100, 3), columns=invalid_col_names))

        assert "The provided feature data is expected to have the following columns:" in str(e.value)
        assert str(tuple(col_names)) in str(e.value)

    @pytest.mark.parametrize("algorithm", ["viterbi", "map"])
    @pytest.mark.parametrize("backend", _trainable_backend_params())
    def test_predict(self, algorithm, backend) -> None:
        model = _new_trainable_hmm(backend)
        model.self_optimize([pd.DataFrame(np.random.rand(100, 3))], [pd.Series(np.random.choice(5, 100))])
        pred = model.predict_hidden_state_sequence(pd.DataFrame(np.random.rand(100, 3)), algorithm=algorithm)
        assert len(pred) == 100
        assert set(pred) == set(range(5))

    @pytest.mark.parametrize("backend", _trainable_backend_params())
    def test_self_optimize_with_info_returns_history(self, backend) -> None:
        data, labels = [pd.DataFrame(np.random.rand(100, 3))], [pd.Series(np.random.choice(5, 100))]
        instance = _new_trainable_hmm(backend)
        trained_instance, history = instance.self_optimize_with_info(data, labels)
        assert instance is trained_instance
        assert history is not None


class TestLegacyBackendHelpers:
    @pytest.mark.parametrize("architecture", ["left-right-strict", "left-right-loose", "fully-connected"])
    def test_different_architectures(self, architecture) -> None:
        legacy = _require_legacy_backend()
        # We test initialization directly, otherwise training will modify the transition matrizes
        model = legacy.initialize_hmm(
            [np.random.rand(100, 3)],
            [np.random.choice(5, 100)],
            n_states=5,
            n_gmm_components=3,
            architecture=architecture,
        )
        transition_matrix = model.dense_transition_matrix()
        expected = np.zeros((7, 7))
        if architecture == "left-right-strict":
            # Normal transitions
            expected[0:5, 0:5] += np.diag(np.ones(5) / 2) + np.diag(np.ones(4) / 2, k=1)
            # Start state
            expected[5, 0] = 1
            # End state
            expected[4, 6] = 0.5
        elif architecture == "left-right-loose":
            # Normal transitions
            expected[0:5, 0:5] += np.diag(np.ones(5) / 3) + np.diag(np.ones(4) / 3, k=1)
            expected[4, 0] = 1 / 3
            # Start state
            expected[5, :5] = 1 / 5
            # End state
            expected[:5, 6] = 1 / 3
        elif architecture == "fully-connected":
            expected[0:5, 0:5] = 1 / 10
            # Start state
            expected[5, :5] = 1 / 5
            # End state
            expected[:5, 6] = 1 / 2
        assert_almost_equal(transition_matrix, expected)


class TestRothSegmentationHmm:
    def test_predict_without_model_raises_error(self) -> None:
        with pytest.raises(ValueError) as e:
            RothSegmentationHmm().predict(pd.DataFrame(np.random.rand(100, 3)), sampling_rate_hz=100)

        assert "No trained model for prediction available!" in str(e.value)

    def test_self_optimize_with_info_returns_history(self) -> None:
        data, labels = (
            [pd.DataFrame(np.random.rand(120, 6), columns=BF_COLS)],
            [_stride_list_to_region_list(pd.DataFrame({"start": [0, 40, 70], "end": [30, 70, 100]}))],
        )
        instance = RothSegmentationHmm(
            hmm_config=_create_roth_hmm_config(stride_n_states=3, stride_n_gmm_components=3)
        ).set_params(
            hmm_config__feature_transform__sampling_rate_feature_space_hz=100,
        )
        trained_instance, history = instance.self_optimize_with_info(data, labels, sampling_rate_hz=100)
        assert instance is trained_instance
        assert isinstance(instance.model, HMMState)
        for v in history.values():
            assert v is not None
        assert set(history.keys()) == {"stride", "transition", "self"}

    def test_serialization_excludes_backend(self) -> None:
        instance = RothSegmentationHmm(hmm_config=_create_roth_hmm_config())

        payload = json.loads(instance.to_json())
        restored = RothSegmentationHmm.from_json(instance.to_json())

        assert set(payload["params"]) == {"hmm_config", "model"}
        assert isinstance(restored.backend, _default_backend_type())

    def test_short_strides_raise_warning(self) -> None:
        data, labels = (
            [pd.DataFrame(np.random.rand(130, 6), columns=BF_COLS)],
            [_stride_list_to_region_list(pd.DataFrame({"start": [0, 40, 70, 110], "end": [30, 70, 100, 114]}))],
        )
        instance = RothSegmentationHmm(
            hmm_config=_create_roth_hmm_config(stride_n_states=5, stride_n_gmm_components=3)
        ).set_params(
            hmm_config__feature_transform__sampling_rate_feature_space_hz=100,
        )
        with pytest.warns(UserWarning) as w:
            instance.self_optimize(data, labels, sampling_rate_hz=100)

        assert any("regions of type `stride`" in str(warning.message) for warning in w)

    def test_unknown_region_type_raises_error(self) -> None:
        data = [pd.DataFrame(np.random.rand(120, 6), columns=BF_COLS)]
        labels = [
            _stride_list_to_region_list(pd.DataFrame({"start": [0, 40, 70], "end": [30, 70, 100]}), "stair_stride")
        ]
        instance = RothSegmentationHmm(hmm_config=_create_roth_hmm_config()).set_params(
            hmm_config__feature_transform__sampling_rate_feature_space_hz=100,
        )

        with pytest.raises(ValidationError) as exc:
            instance.self_optimize(data, labels, sampling_rate_hz=100)

        assert "unknown region types" in str(exc.value)

    def test_missing_configured_module_data_raises_error(self) -> None:
        data = [pd.DataFrame(np.random.rand(120, 6), columns=BF_COLS)]
        labels = [_stride_list_to_region_list(pd.DataFrame({"start": [0, 40, 70], "end": [30, 70, 100]}), "stride")]
        config = CompositeHmmConfig(
            modules=(
                HmmSubModelConfig(
                    name="transition",
                    role="transition",
                    n_states=5,
                    n_gmm_components=3,
                    architecture="left-right-loose",
                ),
                HmmSubModelConfig(
                    name="stride",
                    role="stride",
                    n_states=3,
                    n_gmm_components=3,
                ),
                HmmSubModelConfig(
                    name="stair_stride",
                    role="stride",
                    n_states=3,
                    n_gmm_components=3,
                ),
            )
        )
        instance = RothSegmentationHmm(hmm_config=RothHmmConfig(model_config=config)).set_params(
            hmm_config__feature_transform__sampling_rate_feature_space_hz=100,
        )

        with pytest.raises(ValueError) as exc:
            instance.self_optimize(data, labels, sampling_rate_hz=100)

        assert "stair_stride" in str(exc.value)
        assert "did not receive any trainable regions" in str(exc.value)

    def test_short_transitions_raise_warning(self) -> None:
        data, labels = (
            [pd.DataFrame(np.random.rand(250, 6), columns=BF_COLS)],
            [
                _stride_list_to_region_list(
                    pd.DataFrame({"start": [0, 70, 102, 125, 170], "end": [30, 100, 125, 170, 200]})
                )
            ],
        )
        instance = RothSegmentationHmm(
            hmm_config=_create_roth_hmm_config(
                stride_n_states=5, stride_n_gmm_components=3, transition_n_gmm_components=3
            )
        ).set_params(
            hmm_config__feature_transform__sampling_rate_feature_space_hz=100,
        )
        with pytest.warns(UserWarning) as w:
            instance.self_optimize(data, labels, sampling_rate_hz=100)

        assert any("1 transitions (out of 3)" in str(warning.message) for warning in w)

    def test_strange_inputs_trigger_nan_error(self) -> None:
        # XXXX: We test the skip at the moment because it is not deteministic...
        pytest.skip()

        # I don't understand, why the following inputs trigger the error, but they do.
        # So we use it to test, that the error is raised.
        data, labels = (
            [pd.DataFrame(np.random.rand(200, 6), columns=BF_COLS)],
            [
                _stride_list_to_region_list(
                    pd.DataFrame({"start": [0, 70, 102, 125, 170], "end": [30, 100, 125, 170, 200]})
                )
            ],
        )

        instance = RothSegmentationHmm(
            hmm_config=_create_roth_hmm_config(
                stride_n_states=5, stride_n_gmm_components=3, transition_n_gmm_components=3
            )
        ).set_params(
            hmm_config__feature_transform__sampling_rate_feature_space_hz=100,
        )

        with pytest.warns(UserWarning) as w, pytest.raises(ValueError) as e:
            instance.self_optimize(data, labels, sampling_rate_hz=100)

        assert "During training the improvement per epoch became NaN/infinite or negative!" in str(w[0].message)
        assert "the provided pomegranate model has non-finite/NaN parameters." in str(e.value)

    def test_training_updates_final_model(self) -> None:
        """Training should modify the final fused model while leaving the config untouched."""
        data, labels = (
            [pd.DataFrame(np.random.rand(250, 6), columns=BF_COLS)],
            [
                _stride_list_to_region_list(
                    pd.DataFrame({"start": [0, 70, 102, 125, 170], "end": [30, 100, 125, 170, 200]})
                )
            ],
        )
        instance = RothSegmentationHmm(
            hmm_config=_create_roth_hmm_config(
                stride_n_states=5, stride_n_gmm_components=3, transition_n_gmm_components=3
            )
        ).set_params(
            hmm_config__feature_transform__sampling_rate_feature_space_hz=100,
        )
        hash_model_config = custom_hash(instance.hmm_config)
        hash_model = custom_hash(instance.model)

        instance.self_optimize(data, labels, sampling_rate_hz=100)

        assert hash_model_config == custom_hash(instance.hmm_config)
        assert hash_model != custom_hash(instance.model)

    def test_pretrained_model_is_migrated_to_hmm_state(self) -> None:
        model = PreTrainedRothSegmentationModel()

        assert isinstance(model.model, HMMState)
        assert model.model.trained_with.backend_id == "pomegranate-legacy-migrated"
        assert model.model.trained_with.backend_version is not None
        assert len(model.model.submodels) == 2
        assert isinstance(model.backend, _default_backend_type())

    def test_pretrained_model_migration_removes_silent_backend_states(self) -> None:
        model = PreTrainedRothSegmentationModel()
        compiled = model.model.compiled

        assert compiled.graph.transition_probs.shape == (len(compiled.state_names), len(compiled.state_names))
        assert compiled.graph.start_probs.shape == (len(compiled.state_names),)
        assert compiled.graph.end_probs.shape == (len(compiled.state_names),)
        assert len(compiled.emissions) == len(compiled.state_names)
        assert all(name not in {"start", "end"} for name in compiled.state_names)

    @pytest.mark.parametrize("algorithm", ["map", "viterbi"])
    @pytest.mark.parametrize("sensor", ["left_sensor", "right_sensor"])
    def test_pretrained_scipy_backend_matches_legacy_backend_for_all_decoders(
        self, healthy_example_imu_data, sensor, algorithm
    ) -> None:
        legacy = _require_legacy_backend()
        model = PreTrainedRothSegmentationModel()

        if sensor == "left_sensor":
            data = convert_left_foot_to_fbf(healthy_example_imu_data[sensor])
        else:
            data = convert_to_fbf(healthy_example_imu_data, left_like="left_", right_like="right_")[sensor]

        feature_data, _ = model._transform([data], None, sampling_rate_hz=100)
        feature_data = feature_data[0]

        legacy_result = legacy.backend_class().predict(
            model.model,
            feature_data,
            expected_columns=model.data_columns,
            algorithm=algorithm,
            verbose=model.verbose,
        )
        scipy_result = ScipyHmmInferenceBackend().predict(
            model.model,
            feature_data,
            expected_columns=model.data_columns,
            algorithm=algorithm,
            verbose=model.verbose,
        )

        assert_array_equal(legacy_result, scipy_result)

    def test_pretrained_inference_regression_uses_shared_snapshot(self, healthy_example_imu_data) -> None:
        data = convert_to_fbf(healthy_example_imu_data, left_like="left_", right_like="right_")
        backends = [ScipyHmmInferenceBackend()]
        if PomegranateLegacyHmmBackend is not None:
            backends.append(PomegranateLegacyHmmBackend())
        if PomegranateModernHmmBackend is not None:
            backends.append(PomegranateModernHmmBackend())

        for backend in backends:
            result = HmmStrideSegmentation(
                model=PreTrainedRothSegmentationModel().set_params(backend=backend),
                snap_to_min_win_ms=300,
                snap_to_min_axis="gyr_ml",
            ).segment(data, 204.8)
            for sensor in ["left_sensor", "right_sensor"]:
                expected = load_pretrained_inference_stride_list_snapshot(sensor)
                assert_frame_equal(result.stride_list_[sensor], expected)

    @pytest.mark.parametrize("inference_backend", _runtime_inference_backend_params())
    def test_pretrained_inference_backend_matches_pomegranate_hidden_states(
        self, healthy_example_imu_data, inference_backend
    ) -> None:
        model = PreTrainedRothSegmentationModel()
        comparison_model = model.clone().set_params(backend=inference_backend)
        data = convert_left_foot_to_fbf(healthy_example_imu_data["left_sensor"])

        pomegranate_result = model.predict(data, sampling_rate_hz=100)
        comparison_result = comparison_model.predict(data, sampling_rate_hz=100)

        assert_array_equal(pomegranate_result.hidden_state_sequence_, comparison_result.hidden_state_sequence_)
        assert_array_equal(
            pomegranate_result.hidden_state_sequence_feature_space_,
            comparison_result.hidden_state_sequence_feature_space_,
        )

    def test_trained_model_json_roundtrip_preserves_hidden_states(self) -> None:
        data_sequence = [pd.DataFrame(np.random.rand(120, 6), columns=BF_COLS)]
        region_list_sequence = [_stride_list_to_region_list(pd.DataFrame({"start": [0, 40, 70], "end": [30, 70, 100]}))]
        instance = RothSegmentationHmm(
            hmm_config=_create_roth_hmm_config(stride_n_states=3, stride_n_gmm_components=3)
        ).set_params(
            hmm_config__feature_transform__sampling_rate_feature_space_hz=100,
        )
        instance.self_optimize(data_sequence, region_list_sequence, sampling_rate_hz=100)
        restored = RothSegmentationHmm.from_json(instance.to_json())

        original_result = instance.predict(data_sequence[0], sampling_rate_hz=100)
        restored_result = restored.predict(data_sequence[0], sampling_rate_hz=100)

        assert_array_equal(original_result.hidden_state_sequence_, restored_result.hidden_state_sequence_)
        assert_array_equal(
            original_result.hidden_state_sequence_feature_space_,
            restored_result.hidden_state_sequence_feature_space_,
        )
        assert instance.model.trained_with.backend_id.startswith("pomegranate-")
        assert instance.model.trained_with.backend_version is not None
        assert isinstance(restored.backend, _default_backend_type())

    @pytest.mark.parametrize("inference_backend", _runtime_inference_backend_params())
    def test_trained_inference_backend_matches_pomegranate_hidden_states(self, inference_backend) -> None:
        data_sequence = [pd.DataFrame(np.random.rand(120, 6), columns=BF_COLS)]
        region_list_sequence = [_stride_list_to_region_list(pd.DataFrame({"start": [0, 40, 70], "end": [30, 70, 100]}))]
        instance = RothSegmentationHmm(
            hmm_config=_create_roth_hmm_config(stride_n_states=3, stride_n_gmm_components=3)
        ).set_params(
            hmm_config__feature_transform__sampling_rate_feature_space_hz=100,
        )
        instance.self_optimize(data_sequence, region_list_sequence, sampling_rate_hz=100)

        comparison_instance = instance.clone().set_params(backend=inference_backend)
        pomegranate_result = instance.predict(data_sequence[0], sampling_rate_hz=100)
        comparison_result = comparison_instance.predict(data_sequence[0], sampling_rate_hz=100)

        assert_array_equal(pomegranate_result.hidden_state_sequence_, comparison_result.hidden_state_sequence_)
        assert_array_equal(
            pomegranate_result.hidden_state_sequence_feature_space_,
            comparison_result.hidden_state_sequence_feature_space_,
        )


class TestHmmStrideSegmentation:
    def test_segment_with_single_dataset(self, healthy_example_imu_data) -> None:
        data = convert_left_foot_to_fbf(healthy_example_imu_data["left_sensor"])
        model = PreTrainedRothSegmentationModel()
        instance = HmmStrideSegmentation(model=model)
        result: HmmStrideSegmentation = instance.segment(data, 204.8)

        # PretrainedRothSegmentation is the default model
        assert isinstance(result.result_model_, type(model))
        assert result.result_model_ is not model
        assert isinstance(result.matches_start_end_, np.ndarray)
        assert is_single_sensor_stride_list(result.stride_list_)
        assert isinstance(result.matches_start_end_original_, np.ndarray)
        assert isinstance(result.hidden_state_sequence_, np.ndarray)
        assert result.hidden_state_sequence_ is result.result_model_.hidden_state_sequence_

    @pytest.mark.parametrize("inference_backend", _runtime_inference_backend_params())
    def test_pretrained_inference_backend_matches_pomegranate_segmentation(
        self, healthy_example_imu_data, inference_backend
    ) -> None:
        data = convert_left_foot_to_fbf(healthy_example_imu_data["left_sensor"])
        pomegranate_result = HmmStrideSegmentation(model=PreTrainedRothSegmentationModel()).segment(data, 204.8)
        comparison_result = HmmStrideSegmentation(
            model=PreTrainedRothSegmentationModel().set_params(backend=inference_backend)
        ).segment(data, 204.8)

        assert_array_equal(pomegranate_result.hidden_state_sequence_, comparison_result.hidden_state_sequence_)
        assert_array_equal(pomegranate_result.matches_start_end_, comparison_result.matches_start_end_)
        assert_array_equal(
            pomegranate_result.matches_start_end_original_,
            comparison_result.matches_start_end_original_,
        )

    def test_segment_with_multi_dataset(self, healthy_example_imu_data) -> None:
        data = convert_to_fbf(healthy_example_imu_data, left_like="left_", right_like="right_")
        model = PreTrainedRothSegmentationModel()
        instance = HmmStrideSegmentation(model=model)
        result: HmmStrideSegmentation = instance.segment(data, 204.8)

        assert is_multi_sensor_stride_list(result.stride_list_)

        assert result.result_model_["left_sensor"] is not result.result_model_["right_sensor"]

        for sensor in get_multi_sensor_names(healthy_example_imu_data):
            # PretrainedRothSegmentation is the default model
            assert isinstance(result.result_model_[sensor], type(model))
            assert result.result_model_ is not model
            assert isinstance(result.matches_start_end_[sensor], np.ndarray)
            assert isinstance(result.stride_list_[sensor], pd.DataFrame)
            assert isinstance(result.matches_start_end_original_[sensor], np.ndarray)
            assert isinstance(result.hidden_state_sequence_[sensor], np.ndarray)
            assert result.hidden_state_sequence_[sensor] is result.result_model_[sensor].hidden_state_sequence_

    def test_matches_start_end_and_stride_list_identical(self, healthy_example_imu_data) -> None:
        data = convert_left_foot_to_fbf(healthy_example_imu_data["left_sensor"])[:3000]
        instance = HmmStrideSegmentation()
        result: HmmStrideSegmentation = instance.segment(data, 204.8)

        assert np.array_equal(result.matches_start_end_, result.stride_list_.to_numpy())

    def test_matches_start_end_original_identical_without_post(self, healthy_example_imu_data) -> None:
        data = convert_left_foot_to_fbf(healthy_example_imu_data["left_sensor"])[:3000]

        # With post processing (default), they should be different
        instance = HmmStrideSegmentation()
        result: HmmStrideSegmentation = instance.segment(data, 204.8)

        assert not np.array_equal(result.matches_start_end_original_, result.matches_start_end_)

        # Without post processing, they should be identical
        instance = HmmStrideSegmentation(snap_to_min_win_ms=None)
        result: HmmStrideSegmentation = instance.segment(data, 204.8)

        assert np.array_equal(result.matches_start_end_original_, result.matches_start_end_)

    @pytest.mark.parametrize(
        ("starts", "ends", "correct"),
        [
            # Start at 0, end at end
            (
                np.array([0, 10, 20, 30, 40]),
                np.array([9, 19, 29, 39, 49]),
                np.array([[0, 10], [10, 20], [20, 30], [30, 40], [40, 50]]),
            ),
            # Double states
            (
                np.array([0, 1, 10, 11, 20, 30, 31, 32, 40]),
                np.array([8, 9, 18, 19, 27, 28, 29, 39, 49]),
                np.array([[0, 10], [10, 20], [20, 30], [30, 40], [40, 50]]),
            ),
            # Start at second, end at second to last
            (
                np.array([1, 10, 20, 30, 40]),
                np.array([9, 19, 29, 39, 48]),
                np.array([[1, 10], [10, 20], [20, 30], [30, 40], [40, 49]]),
            ),
            # Last start has no matching end
            (
                np.array([0, 10, 20, 30, 40]),
                np.array([9, 19, 29, 39]),
                np.array([[0, 10], [10, 20], [20, 30], [30, 40]]),
            ),
            # Last start has no matching end 2
            (
                np.array([0, 10, 20, 30, 40]),
                np.array([9, 19, 29]),
                np.array([[0, 10], [10, 20], [20, 30]]),
            ),
            # Multiple starts have same end
            (
                np.array([0, 10, 20, 30, 40]),
                np.array([9, 19, 39, 49, 49]),
                np.array([[0, 10], [10, 20], [20, 40], [40, 50]]),
            ),
            # Ends before start
            (
                np.array([10, 20, 30, 40]),
                np.array([9, 19, 39, 49, 49]),
                np.array([[10, 20], [20, 40], [40, 50]]),
            ),
        ],
    )
    def test_hidden_state_sequence_start_end(self, starts, ends, correct) -> None:
        """Test that the start end values are correctly extracted."""
        hidden_state_sequence = np.zeros(50)
        hidden_state_sequence[starts] = 1
        hidden_state_sequence[ends] = 2

        starts_ends = HmmStrideSegmentation()._hidden_states_to_matches_start_end(hidden_state_sequence, 1, 2)

        assert_array_equal(starts_ends, correct)


def test_pre_trained_model_returns_correctly() -> None:
    assert isinstance(PreTrainedRothSegmentationModel(), RothSegmentationHmm)
