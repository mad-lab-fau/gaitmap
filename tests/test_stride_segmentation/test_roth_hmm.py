import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from gaitmap.data_transform import SlidingWindowMean
from gaitmap.stride_segmentation.hmm import (
    HmmStrideSegmentation,
    PreTrainedRothSegmentationModel,
    RothHmmFeatureTransformer,
    RothSegmentationHmm,
    SimpleHmm,
)
from gaitmap.utils.consts import BF_COLS
from gaitmap.utils.coordinate_conversion import convert_left_foot_to_fbf, convert_to_fbf
from gaitmap.utils.datatype_helper import (
    get_multi_sensor_names,
    is_multi_sensor_stride_list,
    is_single_sensor_stride_list,
)
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

    def test_empty_init(self) -> None:
        pytest.skip()


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


class TestSimpleModel:
    def test_error_on_different_number_data_and_labels(self) -> None:
        with pytest.raises(ValueError) as e:
            SimpleHmm(n_states=5, n_gmm_components=3).self_optimize(
                [np.random.rand(100, 3)], [np.random.rand(100), np.random.rand(100)]
            )

        assert "The given training sequence and initial training labels" in str(e.value)

    def test_error_if_datasequence_shorter_nstates(self) -> None:
        with pytest.raises(ValueError) as e:
            SimpleHmm(n_states=5, n_gmm_components=3).self_optimize(
                [np.random.rand(100, 3), np.random.rand(3, 3)], [np.random.rand(100), np.random.rand(3)]
            )

        assert "Invalid training sequence!" in str(e.value)

    def test_error_on_different_length_data_and_labels(self) -> None:
        with pytest.raises(ValueError) as e:
            SimpleHmm(n_states=5, n_gmm_components=3).self_optimize(
                [pd.DataFrame(np.random.rand(100, 3))], [pd.Series(np.random.rand(99))]
            )

        assert "a different number of samples" in str(e.value)

    def test_invalid_label_sequence(self) -> None:
        n_states = 5
        with pytest.raises(ValueError) as e:
            SimpleHmm(n_states=n_states, n_gmm_components=3).self_optimize(
                [pd.DataFrame(np.random.rand(100, 3))], [pd.Series(np.full(100, n_states + 1))]
            )

        assert "Invalid label sequence" in str(e.value)

    def test_model_exists_warning(self) -> None:
        model = SimpleHmm(n_states=5, n_gmm_components=3)
        model.self_optimize([pd.DataFrame(np.random.rand(100, 3))], [pd.Series(np.random.choice(5, 100))])
        with pytest.warns(UserWarning) as e:
            model.self_optimize([pd.DataFrame(np.random.rand(100, 3))], [pd.Series(np.random.choice(5, 100))])

        assert "Model already exists" in str(e[0].message)

    def test_predict_rasies_error_without_optimize(self) -> None:
        with pytest.raises(ValueError) as e:
            SimpleHmm(n_states=5, n_gmm_components=3).predict_hidden_state_sequence(
                pd.DataFrame(np.random.rand(100, 3))
            )

        assert "You need to train the HMM before calling `predict_hidden_state_sequence`" in str(e.value)

    def test_predict_raises_error_on_invalid_columns(self) -> None:
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
    def test_predict(self, algorithm) -> None:
        model = SimpleHmm(n_states=5, n_gmm_components=3)
        model.self_optimize([pd.DataFrame(np.random.rand(100, 3))], [pd.Series(np.random.choice(5, 100))])
        pred = model.predict_hidden_state_sequence(pd.DataFrame(np.random.rand(100, 3)), algorithm=algorithm)
        assert len(pred) == 100
        assert set(pred) == set(range(5))

    @pytest.mark.parametrize(
        ("architecture", "transitions", "starts", "ends"),
        [
            ("left-right-strict", [[0.5, 0.5], [0.0, 0.5]], [1.0, 0.0], [0.0, 0.5]),
            ("left-right-loose", [[1 / 3, 1 / 3], [1 / 3, 1 / 3]], [0.5, 0.5], [1 / 3, 1 / 3]),
            ("fully-connected", [[0.25, 0.25], [0.25, 0.25]], [0.5, 0.5], [0.5, 0.5]),
        ],
    )
    def test_different_architectures(self, architecture, transitions, starts, ends) -> None:
        """Public initialization normalizes permitted transitions together with termination."""
        data = pd.DataFrame({"feature": [-2.0, -1.0, 1.0, 2.0]})
        labels = pd.Series([0, 0, 1, 1])
        result = SimpleHmm(2, 1, architecture=architecture, max_iterations=0, verbose=False).self_optimize(
            [data], [labels]
        )

        assert_allclose(result.model.transitions, transitions)
        assert_allclose(result.model.starts, starts)
        assert_allclose(result.model.ends, ends)

    def test_invalid_architecture_raises_error(self) -> None:
        with pytest.raises(ValueError) as e:
            SimpleHmm(n_states=5, n_gmm_components=3, architecture="invalid").self_optimize(
                [pd.DataFrame(np.random.rand(100, 3))], [pd.Series(np.random.choice(5, 100))]
            )

        assert "Invalid architecture" in str(e.value)


class TestRothSegmentationHmm:
    def test_predict_without_model_raises_error(self) -> None:
        with pytest.raises(ValueError) as e:
            RothSegmentationHmm().predict(pd.DataFrame(np.random.rand(100, 3)), sampling_rate_hz=100)

        assert "No trained model for prediction available!" in str(e.value)

    def test_short_strides_raise_warning(self) -> None:
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

    def test_short_transitions_raise_warning(self) -> None:
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

        assert any("1 transitions (out of 3)" in str(warning.message) for warning in w)

    def test_strange_inputs_trigger_nan_error(self) -> None:
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
        assert "the provided HMM model has non-finite/NaN parameters." in str(e.value)


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
