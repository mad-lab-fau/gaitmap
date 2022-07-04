from importlib.resources import open_text

import joblib
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from pandas._testing import assert_frame_equal

from gaitmap.data_transform import FixedScaler, IdentityTransformer, TrainableTransformerMixin
from gaitmap.stride_segmentation import BarthOriginalTemplate, DtwTemplate, InterpolatedDtwTemplate
from gaitmap.utils.exceptions import ValidationError
from tests.conftest import compare_algo_objects


class TestSerialize:
    """Test that templates can be serialized correctly."""

    @pytest.mark.parametrize("dtype", (list, np.array, pd.Series, pd.DataFrame))
    def test_different_dtypes(self, dtype):
        template = dtype(list(range(10)))

        instance = DtwTemplate(data=template)

        new_instance = DtwTemplate.from_json(instance.to_json())

        compare_algo_objects(instance, new_instance)

    def test_index_order_long_dfs(self):
        """Loading df based templates might change their index."""
        template = pd.DataFrame(list(range(20)))

        instance = DtwTemplate(data=template)

        new_instance = DtwTemplate.from_json(instance.to_json())

        compare_algo_objects(instance, new_instance)


class TestTemplateBaseClass:
    def test_template_provided(self):
        """Test very simple case where the template is directly stored in the class instance."""
        template = np.arange(10)

        instance = DtwTemplate(data=template)

        assert_array_equal(instance.get_data(), template)

    def test_no_valid_info_provided(self):
        """Test that an error is raised, if neither a filename nor a array is provided."""
        instance = DtwTemplate()

        with pytest.raises(ValueError):
            _ = instance.get_data()

    def test_use_columns_array(self):
        template = np.stack((np.arange(10), np.arange(10, 20))).T

        instance = DtwTemplate(data=template, use_cols=[1])

        assert_array_equal(instance.get_data(), template[:, 1])

    def test_use_columns_dataframe(self):
        template = np.stack((np.arange(10), np.arange(10, 20))).T
        template = pd.DataFrame(template, columns=["col_1", "col_2"])

        instance = DtwTemplate(data=template, use_cols=["col_1"])

        assert_array_equal(instance.get_data(), template[["col_1"]])

    def test_use_columns_wrong_dim(self):
        template = np.arange(10)

        instance = DtwTemplate(data=template, use_cols=[1])

        with pytest.raises(ValueError):
            _ = instance.get_data()

    def test_get_data_applies_scaling(self):
        template = pd.DataFrame(np.arange(10))

        instance = DtwTemplate(data=template, scaling=FixedScaler(scale=2))

        assert_array_equal(instance.get_data(), template / 2)


class TestBartTemplate:
    def test_load(self):
        with open_text("gaitmap_mad.stride_segmentation._dtw_templates", "barth_original_template.csv") as test_data:
            data = pd.read_csv(test_data, header=0)

        barth_instance = BarthOriginalTemplate()

        assert_frame_equal(barth_instance.get_data(), data / 500.0)
        assert barth_instance.sampling_rate_hz == 204.8
        assert barth_instance.scaling.get_params() == FixedScaler(500.0, 0).get_params()

    def test_hashing(self):
        """Test that calling `get_data` does not modify the hash of the object."""
        barth_instance = BarthOriginalTemplate()

        before_hash = joblib.hash(barth_instance)
        barth_instance.get_data()
        after_hash = joblib.hash(barth_instance)

        assert before_hash == after_hash


class TestCreateTemplate:
    def test_create_template_simple(self):
        template = np.arange(10)
        sampling_rate_hz = 100

        instance = DtwTemplate(data=template, sampling_rate_hz=sampling_rate_hz)

        assert_array_equal(instance.get_data(), template)
        assert instance.sampling_rate_hz == sampling_rate_hz

    def test_create_template_use_col(self):
        template = np.stack((np.arange(10), np.arange(10, 20))).T
        template = pd.DataFrame(template, columns=["col_1", "col_2"])
        sampling_rate_hz = 100
        use_cols = ("col_1",)

        instance = DtwTemplate(data=template, sampling_rate_hz=sampling_rate_hz, use_cols=use_cols)

        assert_array_equal(instance.get_data(), template[["col_1"]])
        assert instance.sampling_rate_hz == sampling_rate_hz
        assert_array_equal(instance.use_cols, use_cols)


class TestCreateInterpolatedTemplate:
    @pytest.fixture(autouse=True, params=["linear", "nearest"])
    def select_kind(self, request):
        self.kind = request.param

    def test_create_interpolated_template_single_dataset(self):
        """Test function can handle single dataset input."""
        template_data = pd.DataFrame(np.array([0, 1, 2, 1, 0]), columns=["dummy_col"])
        instance = InterpolatedDtwTemplate(interpolation_method=self.kind, n_samples=None).self_optimize(
            [template_data]
        )

        assert_array_almost_equal(instance.get_data(), template_data[["dummy_col"]])
        assert isinstance(instance, DtwTemplate)

    def test_create_interpolated_template_dataset_list(self):
        """Test function can handle lists of dataset input."""
        template_data1 = pd.DataFrame(np.array([0, 1, 2, 1, 0]), columns=["dummy_col"])
        template_data2 = pd.DataFrame(np.array([0, 1, 2, 1, 0]), columns=["dummy_col"])

        instance = InterpolatedDtwTemplate(interpolation_method=self.kind, n_samples=None).self_optimize(
            [template_data1, template_data2],
            sampling_rate_hz=1,
        )

        assert_array_almost_equal(instance.get_data(), template_data1[["dummy_col"]])
        assert isinstance(instance, DtwTemplate)

    def test_create_interpolated_template_different_indices(self):
        """Test that interpolation works even if strides have different indices."""
        template_data1 = pd.DataFrame(np.array([0, 1, 2, 1, 0]), columns=["dummy_col"])
        template_data2 = pd.DataFrame(np.array([0, 1, 2, 1, 0]), columns=["dummy_col"])
        template_data2.index = template_data2.index + 10

        instance = InterpolatedDtwTemplate(interpolation_method=self.kind, n_samples=None).self_optimize(
            [template_data1, template_data2],
            sampling_rate_hz=1,
        )

        assert_array_almost_equal(instance.get_data(), template_data1[["dummy_col"]])
        assert isinstance(instance, DtwTemplate)

    def test_create_interpolated_template_calculates_mean(self):
        """Test if result is actually mean over all inputs."""
        template_data1 = pd.DataFrame(np.array([0, 1, 2, 1, 0]), columns=["dummy_col"])
        template_data2 = pd.DataFrame(np.array([0, -1, -2, -1, 0]), columns=["dummy_col"])

        instance = InterpolatedDtwTemplate(interpolation_method=self.kind, n_samples=None).self_optimize(
            [template_data1, template_data2],
            sampling_rate_hz=1,
        )

        result_template_df = pd.DataFrame(np.array([0, 0, 0, 0, 0]), columns=["dummy_col"])
        assert_array_almost_equal(instance.get_data(), result_template_df.to_numpy())
        assert isinstance(instance, DtwTemplate)

    def test_create_interpolated_mean_length_over_input_sequences_template(self):
        """Test template has mean length of all input sequences."""
        template_data1 = pd.DataFrame(np.array([0, 1, 2, 3, 4]), columns=["dummy_col"])
        template_data2 = pd.DataFrame(np.array([0, 1, 2]), columns=["dummy_col"])

        instance = InterpolatedDtwTemplate(interpolation_method=self.kind, n_samples=None).self_optimize(
            [template_data1, template_data2],
            sampling_rate_hz=1,
        )

        assert len(instance.get_data()) == 4
        assert instance.sampling_rate_hz == 1
        assert isinstance(instance, DtwTemplate)

    def test_create_interpolated_fixed_length_template_upsample(self):
        """Test template has specified length for upsampling."""
        template_data1 = pd.DataFrame(np.array([0, 1, 2, 3, 4]), columns=["dummy_col"])
        template_data2 = pd.DataFrame(np.array([0, 1, 2]), columns=["dummy_col"])

        instance = InterpolatedDtwTemplate(interpolation_method=self.kind, n_samples=5).self_optimize(
            [template_data1, template_data2],
            sampling_rate_hz=1,
        )
        assert len(instance.get_data()) == 5
        # Effective sampling rate is calculated based on the number of samples before and after interpolation
        assert instance.sampling_rate_hz == 5 / 4
        assert isinstance(instance, DtwTemplate)

    def test_create_interpolated_fixed_length_template_downsample(self):
        """Test template has specified length for downsampling."""
        template_data1 = pd.DataFrame(np.array([0, 1, 2, 3, 5]), columns=["dummy_col"])
        template_data2 = pd.DataFrame(np.array([0, 1, 2, 3, 4, 5, 6]), columns=["dummy_col"])

        instance = InterpolatedDtwTemplate(interpolation_method=self.kind, n_samples=3).self_optimize(
            [template_data1, template_data2],
            sampling_rate_hz=1,
        )
        assert_array_equal(len(instance.get_data()), 3)
        # Effective sampling rate is calculated based on the number of samples before and after interpolation
        assert instance.sampling_rate_hz == 3 / 6
        assert isinstance(instance, DtwTemplate)

    def test_create_interpolated_template_check_multisensordataset_exception(self):
        """Test only single sensor datasets are valid input."""
        template_data1 = pd.DataFrame(np.array([[0, 1, 2], [0, 1, 2]]), columns=["col_1", "col_2", "col_3"])
        template_data2 = pd.DataFrame(np.array([[0, 1, 2], [0, 1, 2]]), columns=["col_1", "col_2", "col_3"])

        dataset = {"left_sensor": template_data1, "right_sensor": template_data2}

        with pytest.raises(ValidationError, match=r".* SingleSensorData*"):
            InterpolatedDtwTemplate().self_optimize(dataset, kind=self.kind, n_samples=None)

    def test_scaling_retraining(self):
        class CustomScaler(IdentityTransformer, TrainableTransformerMixin):
            """Dummy scaler that records the data it is trained with."""

            def __init__(self, opti_data=None):
                self.opti_data = opti_data

            def self_optimize(self, data, **_):
                self.opti_data = data
                return self

        template_data1 = pd.DataFrame(np.array([0, 1, 2, 1, 0]), columns=["dummy_col"])
        template_data2 = pd.DataFrame(np.array([0, -1, -2, -1, 0]), columns=["dummy_col"])

        scaler_instance = CustomScaler()

        instance = InterpolatedDtwTemplate(
            interpolation_method=self.kind, n_samples=None, scaling=scaler_instance
        ).self_optimize(
            [template_data1, template_data2],
            sampling_rate_hz=1,
        )

        assert_array_equal(scaler_instance.opti_data[0], instance.data)

    def test_column_selection(self):
        data1 = pd.DataFrame(np.ones((5, 3)), columns=["col_1", "col_2", "col_3"])
        data2 = pd.DataFrame(np.ones((5, 3)), columns=["col_2", "col_1", "col_3"])

        instance = InterpolatedDtwTemplate().self_optimize(
            [data1, data2], sampling_rate_hz=1, columns=["col_3", "col_1"]
        )

        assert instance.get_data().columns.tolist() == ["col_3", "col_1"]
