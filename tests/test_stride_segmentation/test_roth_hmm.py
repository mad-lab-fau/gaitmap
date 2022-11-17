from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pomegranate import GeneralMixtureModel
from pomegranate.hmm import History

from gaitmap.data_transform import SlidingWindowMean
from gaitmap.utils.coordinate_conversion import convert_left_foot_to_fbf
from gaitmap_mad.stride_segmentation.hmm import (
    HmmStrideSegmentation,
    PreTrainedRothSegmentationModel,
    RothHmmFeatureTransformer,
    RothSegmentationHmm,
    SimpleHmm,
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

    def test_empty_init(self):
        pytest.skip()


class TestRothHmmFeatureTransform:
    @pytest.mark.parametrize("target_sampling_rate", [50, 25])
    def test_inverse_transform_state_sequence(self, target_sampling_rate):
        transform = RothHmmFeatureTransformer(sampling_frequency_feature_space_hz=target_sampling_rate)
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
        assert set(transformed_data.columns) == set(
            f"{feature}{feature_prefixes[feature]}__{axis}" for feature in features for axis in axes
        )

    def test_actual_output(self, healthy_example_imu_data):
        # We disable downsampling, standardization, and filtering for this test
        transform = RothHmmFeatureTransformer(
            sampling_frequency_feature_space_hz=100,
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

    @pytest.mark.parametrize("roi, data", [(None, []), ([], None)])
    def test_value_error_missing_sampling_rate(self, roi, data):
        with pytest.raises(ValueError) as e:
            RothHmmFeatureTransformer().transform(data, roi_list=roi, sampling_rate_hz=None)

        assert "sampling_rate_hz" in str(e.value)

    def test_resample_roi(self):
        transform = RothHmmFeatureTransformer(sampling_frequency_feature_space_hz=50)
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
                assert set(d.name for d in dists) == {"MultivariateGaussianDistribution"}

    def test_model_exists_warning(self):
        model = SimpleHmm(n_states=5, n_gmm_components=3)
        model.self_optimize([pd.DataFrame(np.random.rand(100, 3))], [pd.Series(np.random.choice(5, 100))])
        with pytest.warns(UserWarning) as e:
            model.self_optimize([pd.DataFrame(np.random.rand(100, 3))], [pd.Series(np.random.choice(5, 100))])

        assert "Model already exists" in str(e[0].message)

    def test_predict(self):
        # TODO: Test predict
        pass

    def test_different_architectures(self):
        # TODO: Test different architectures
        pass

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
