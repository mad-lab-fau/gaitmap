from unittest.mock import patch

import pytest

pytest.importorskip("pomegranate")

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from pomegranate import GeneralMixtureModel
from pomegranate.hmm import History
from tpcp._hash import custom_hash

from gaitmap.data_transform import SlidingWindowMean
from gaitmap.utils.consts import BF_COLS
from gaitmap.utils.coordinate_conversion import convert_left_foot_to_fbf, convert_to_fbf
from gaitmap.utils.datatype_helper import (
    get_multi_sensor_names,
    is_multi_sensor_stride_list,
    is_single_sensor_stride_list,
)
from gaitmap_mad.stride_segmentation.hmm import (
    HmmStrideSegmentation,
    PreTrainedRothSegmentationModel,
    RothHmmFeatureTransformer,
    RothSegmentationHmm,
    SimpleHmm,
)
from gaitmap_mad.stride_segmentation.hmm._simple_model import initialize_hmm
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin

# Fix random seed for reproducibility
np.random.seed(1)


class TestMetaFunctionalityRothSegmentationHmm(TestAlgorithmMixin):
    __test__ = True

    algorithm_class = RothSegmentationHmm

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data) -> RothSegmentationHmm:
        hmm = PreTrainedRothSegmentationModel()
        hmm.predict(convert_left_foot_to_fbf(healthy_example_imu_data["left_sensor"]), sampling_rate_hz=100)
        return hmm


class TestMetaFunctionalityHmmStrideSegmentation(TestAlgorithmMixin):
    __test__ = True

    algorithm_class = HmmStrideSegmentation

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data) -> HmmStrideSegmentation:
        hmm = HmmStrideSegmentation()
        hmm.segment(convert_left_foot_to_fbf(healthy_example_imu_data["left_sensor"]), sampling_rate_hz=100)
        return hmm


class TestMetaFunctionalityRothHMMFeatureTransformer(TestAlgorithmMixin):
    __test__ = True

    algorithm_class = RothHmmFeatureTransformer

    @pytest.fixture()
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


class TestMetaFunctionalitySimpleHMM(TestAlgorithmMixin):
    __test__ = True

    algorithm_class = SimpleHmm

    @pytest.fixture()
    def valid_instance(self, after_action_instance):
        return SimpleHmm(n_states=5, n_gmm_components=3)

    def test_empty_init(self):
        pytest.skip()


class TestRothHmmFeatureTransform:
    @pytest.mark.parametrize("target_sampling_rate", [50, 25])
    def test_inverse_transform_state_sequence(self, target_sampling_rate):
        transform = RothHmmFeatureTransformer(sampling_rate_feature_space_hz=target_sampling_rate)
        in_state_sequence = np.array([0, 1, 2, 3, 4, 5])
        state_sequence = transform.inverse_transform_state_sequence(
            state_sequence=in_state_sequence,
            sampling_rate_hz=100,
        )
        # output should have the same sampling rate, by repeating values
        assert len(state_sequence) == 100 / target_sampling_rate * len(in_state_sequence)
        assert_array_equal(state_sequence, np.repeat(in_state_sequence, 100 / target_sampling_rate))

    @pytest.mark.parametrize("features", [["raw"], ["raw", "gradient"], ["raw", "gradient", "mean"]])
    @pytest.mark.parametrize("axes", [["gyr_ml"], ["acc_pa"], ["gyr_ml", "acc_pa"]])
    def test_select_features(self, features, healthy_example_imu_data, axes):
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

    def test_actual_output(self, healthy_example_imu_data):
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

    def test_type_error_filter(self):
        with pytest.raises(TypeError) as e:
            RothHmmFeatureTransformer(low_pass_filter="test").transform([], sampling_rate_hz=100)

        assert "low_pass_filter" in str(e.value)

    @pytest.mark.parametrize(("roi", "data"), [(None, []), ([], None)])
    def test_value_error_missing_sampling_rate(self, roi, data):
        with pytest.raises(ValueError) as e:
            RothHmmFeatureTransformer().transform(data, roi_list=roi, sampling_rate_hz=None)

        assert "sampling_rate_hz" in str(e.value)

    def test_resample_roi(self):
        transform = RothHmmFeatureTransformer(sampling_rate_feature_space_hz=50)
        roi = pd.DataFrame(np.array([[0, 100], [200, 300], [400, 500]]), columns=["start", "end"])
        resampled_roi = transform.transform(roi_list=roi, sampling_rate_hz=100).transformed_roi_list_
        assert_array_equal(resampled_roi, np.array([[0, 50], [100, 150], [200, 250]]))
        assert transform.roi_list is roi
        assert transform.sampling_rate_hz == 100


class TestSimpleModel:
    def test_error_on_different_number_data_and_labels(self):
        with pytest.raises(ValueError) as e:
            SimpleHmm(n_states=5, n_gmm_components=3).self_optimize(
                [np.random.rand(100, 3)], [np.random.rand(100), np.random.rand(100)]
            )

        assert "The given training sequence and initial training labels" in str(e.value)

    def test_error_if_datasequence_shorter_nstates(self):
        with pytest.raises(ValueError) as e:
            SimpleHmm(n_states=5, n_gmm_components=3).self_optimize(
                [np.random.rand(100, 3), np.random.rand(3, 3)], [np.random.rand(100), np.random.rand(3)]
            )

        assert "Invalid training sequence!" in str(e.value)

    def test_error_on_different_length_data_and_labels(self):
        with pytest.raises(ValueError) as e:
            SimpleHmm(n_states=5, n_gmm_components=3).self_optimize(
                [pd.DataFrame(np.random.rand(100, 3))], [pd.Series(np.random.rand(99))]
            )

        assert "a different number of samples" in str(e.value)

    def test_invalid_label_sequence(self):
        n_states = 5
        with pytest.raises(ValueError) as e:
            SimpleHmm(n_states=n_states, n_gmm_components=3).self_optimize(
                [pd.DataFrame(np.random.rand(100, 3))], [pd.Series(np.full(100, n_states + 1))]
            )

        assert "Invalid label sequence" in str(e.value)

    @pytest.mark.parametrize(
        "data",
        [
            pd.DataFrame(np.random.rand(100, 3), columns=["feature1", "feature2", "feature3"]),
            pd.DataFrame(np.random.rand(100, 1), columns=["feature1"]),
        ],
    )
    @pytest.mark.parametrize("n_gmm_components", [1, 3])
    # We test one value with n_states > 10, as this should trigger a sorting bug in pomegranate that we are handling
    # explicitly
    @pytest.mark.parametrize("n_states", [5, 12])
    def test_optimize_with_single_sequence(self, data, n_gmm_components, n_states):
        model = SimpleHmm(n_states=n_states, n_gmm_components=n_gmm_components, max_iterations=1)
        model.self_optimize([data], [pd.Series(np.tile(np.arange(n_states), int(np.ceil(100 / n_states)))[:100])])

        assert list(model.data_columns) == data.columns.tolist()
        # -2 because of the start and end state
        assert len(model.model.states) - 2 == n_states == model.n_states
        # Test that each state has 3 gmm components
        for state in model.model.states:
            if state.name not in ["None-start", "None-end"]:
                if n_gmm_components == 1:
                    dists = [state.distribution]
                else:
                    assert isinstance(state.distribution, GeneralMixtureModel)
                    assert len(state.distribution.distributions) == model.n_gmm_components == n_gmm_components
                    dists = state.distribution.distributions
                assert {d.name for d in dists} == {"MultivariateGaussianDistribution"}

    def test_model_exists_warning(self):
        model = SimpleHmm(n_states=5, n_gmm_components=3)
        model.self_optimize([pd.DataFrame(np.random.rand(100, 3))], [pd.Series(np.random.choice(5, 100))])
        with pytest.warns(UserWarning) as e:
            model.self_optimize([pd.DataFrame(np.random.rand(100, 3))], [pd.Series(np.random.choice(5, 100))])

        assert "Model already exists" in str(e[0].message)

    def test_predict_rasies_error_without_optimize(self):
        with pytest.raises(ValueError) as e:
            SimpleHmm(n_states=5, n_gmm_components=3).predict_hidden_state_sequence(
                pd.DataFrame(np.random.rand(100, 3))
            )

        assert "You need to train the HMM before calling `predict_hidden_state_sequence`" in str(e.value)

    def test_predict_raises_error_on_invalid_columns(self):
        model = SimpleHmm(n_states=5, n_gmm_components=3)
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
    def test_predict(self, algorithm):
        model = SimpleHmm(n_states=5, n_gmm_components=3)
        model.self_optimize([pd.DataFrame(np.random.rand(100, 3))], [pd.Series(np.random.choice(5, 100))])
        pred = model.predict_hidden_state_sequence(pd.DataFrame(np.random.rand(100, 3)), algorithm=algorithm)
        assert len(pred) == 100
        assert set(pred) == set(range(5))

    @pytest.mark.parametrize("architecture", ["left-right-strict", "left-right-loose", "fully-connected"])
    def test_different_architectures(self, architecture):
        # We test initialization directly, otherwise training will modify the transition matrizes
        model = initialize_hmm(
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

    def test_self_optimize_calls_self_optimize_with_info(self):
        data, labels = [pd.DataFrame(np.random.rand(100, 3))], [pd.Series(np.random.choice(5, 100))]

        with patch.object(SimpleHmm, "self_optimize_with_info") as mock:
            instance = SimpleHmm(n_states=5, n_gmm_components=3)
            mock.return_value = (instance, None)
            instance.self_optimize(data, labels)

            mock.assert_called_once_with(data, labels)

    def test_self_optimize_with_info_returns_history(self):
        data, labels = [pd.DataFrame(np.random.rand(100, 3))], [pd.Series(np.random.choice(5, 100))]
        instance = SimpleHmm(n_states=5, n_gmm_components=3)
        trained_instance, history = instance.self_optimize_with_info(data, labels)
        assert instance is trained_instance
        assert isinstance(history, History)

    def test_invalid_architecture_raises_error(self):
        with pytest.raises(ValueError) as e:
            SimpleHmm(n_states=5, n_gmm_components=3, architecture="invalid").self_optimize(
                [pd.DataFrame(np.random.rand(100, 3))], [pd.Series(np.random.choice(5, 100))]
            )

        assert "Invalid architecture" in str(e.value)


class TestRothSegmentationHmm:
    def test_predict_without_model_raises_error(self):
        with pytest.raises(ValueError) as e:
            RothSegmentationHmm().predict(pd.DataFrame(np.random.rand(100, 3)), sampling_rate_hz=100)

        assert "No trained model for prediction available!" in str(e.value)

    def test_self_optimize_calls_self_optimize_with_info(self):
        data, labels = [pd.DataFrame(np.random.rand(100, 3))], [pd.DataFrame({"start": [0], "end": [100]})]

        with patch.object(RothSegmentationHmm, "self_optimize_with_info") as mock:
            instance = RothSegmentationHmm()
            mock.return_value = (instance, None)
            instance.self_optimize(data, labels, sampling_rate_hz=100)

            mock.assert_called_once_with(data, labels, sampling_rate_hz=100)

    def test_self_optimize_with_info_returns_history(self):
        data, labels = (
            [pd.DataFrame(np.random.rand(120, 6), columns=BF_COLS)],
            [pd.DataFrame({"start": [0, 40, 70], "end": [30, 70, 100]})],
        )
        instance = RothSegmentationHmm().set_params(
            feature_transform__sampling_rate_feature_space_hz=100,
            stride_model__n_states=3,
            stride_model__n_gmm_components=3,
        )
        trained_instance, history = instance.self_optimize_with_info(data, labels, sampling_rate_hz=100)
        assert instance is trained_instance
        for v in history.values():
            assert isinstance(v, History)
        assert set(history.keys()) == {"stride_model", "transition_model", "self"}

    def test_short_strides_raise_warning(self):
        data, labels = (
            [pd.DataFrame(np.random.rand(130, 6), columns=BF_COLS)],
            [pd.DataFrame({"start": [0, 40, 70, 110], "end": [30, 70, 100, 114]})],
        )
        instance = RothSegmentationHmm().set_params(
            feature_transform__sampling_rate_feature_space_hz=100,
            stride_model__n_states=5,
            stride_model__n_gmm_components=3,
        )
        with pytest.warns(UserWarning) as w:
            instance.self_optimize(data, labels, sampling_rate_hz=100)

        assert "1 strides (out of 4)" in str(w[0].message)

    def test_short_transitions_raise_warning(self):
        data, labels = (
            [pd.DataFrame(np.random.rand(250, 6), columns=BF_COLS)],
            [pd.DataFrame({"start": [0, 70, 102, 125, 170], "end": [30, 100, 125, 170, 200]})],
        )
        instance = RothSegmentationHmm().set_params(
            feature_transform__sampling_rate_feature_space_hz=100,
            transition_model__n_gmm_components=3,
            stride_model__n_gmm_components=3,
            stride_model__n_states=5,
        )
        with pytest.warns(UserWarning) as w:
            instance.self_optimize(data, labels, sampling_rate_hz=100)

        # The first warning is the warning about negative improvements during training
        assert "1 transitions (out of 3)" in str(w[1].message)

    def test_strange_inputs_trigger_nan_error(self):
        # XXXX: We test the skip at the moment because it is not deteministic...
        pytest.skip()

        # I don't understand, why the following inputs trigger the error, but they do.
        # So we use it to test, that the error is raised.
        data, labels = (
            [pd.DataFrame(np.random.rand(200, 6), columns=BF_COLS)],
            [pd.DataFrame({"start": [0, 70, 102, 125, 170], "end": [30, 100, 125, 170, 200]})],
        )

        instance = RothSegmentationHmm().set_params(
            feature_transform__sampling_rate_feature_space_hz=100,
            transition_model__n_gmm_components=3,
            stride_model__n_gmm_components=3,
            stride_model__n_states=5,
        )

        with pytest.warns(UserWarning) as w, pytest.raises(ValueError) as e:
            instance.self_optimize(data, labels, sampling_rate_hz=100)

        assert "During training the improvement per epoch became NaN/infinite or negative!" in str(w[0].message)
        assert "the provided pomegranate model has non-finite/NaN parameters." in str(e.value)

    def test_training_updates_all_models(self):
        """Training should modify the stride, the transition model and the model itself."""
        data, labels = (
            [pd.DataFrame(np.random.rand(250, 6), columns=BF_COLS)],
            [pd.DataFrame({"start": [0, 70, 102, 125, 170], "end": [30, 100, 125, 170, 200]})],
        )
        instance = RothSegmentationHmm().set_params(
            feature_transform__sampling_rate_feature_space_hz=100,
            transition_model__n_gmm_components=3,
            stride_model__n_gmm_components=3,
            stride_model__n_states=5,
        )
        # We can not properly test this, so we get the hash before and after and compare them.
        hash_stride_model = custom_hash(instance.stride_model)
        hash_transition_model = custom_hash(instance.transition_model)
        hash_model = custom_hash(instance.model)

        instance.self_optimize(data, labels, sampling_rate_hz=100)

        assert hash_stride_model != custom_hash(instance.stride_model)
        assert hash_transition_model != custom_hash(instance.transition_model)
        assert hash_model != custom_hash(instance.model)


class TestHmmStrideSegmentation:
    def test_segment_with_single_dataset(self, healthy_example_imu_data):
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

    def test_segment_with_multi_dataset(self, healthy_example_imu_data):
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

    def test_matches_start_end_and_stride_list_identical(self, healthy_example_imu_data):
        data = convert_left_foot_to_fbf(healthy_example_imu_data["left_sensor"])[:3000]
        instance = HmmStrideSegmentation()
        result: HmmStrideSegmentation = instance.segment(data, 204.8)

        assert np.array_equal(result.matches_start_end_, result.stride_list_.to_numpy())

    def test_matches_start_end_original_identical_without_post(self, healthy_example_imu_data):
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
    def test_hidden_state_sequence_start_end(self, starts, ends, correct):
        """Test that the start end values are correctly extracted."""
        hidden_state_sequence = np.zeros(50)
        hidden_state_sequence[starts] = 1
        hidden_state_sequence[ends] = 2

        starts_ends = HmmStrideSegmentation()._hidden_states_to_matches_start_end(hidden_state_sequence, 1, 2)

        assert_array_equal(starts_ends, correct)


def test_pre_trained_model_returns_correctly():
    assert isinstance(PreTrainedRothSegmentationModel(), RothSegmentationHmm)
