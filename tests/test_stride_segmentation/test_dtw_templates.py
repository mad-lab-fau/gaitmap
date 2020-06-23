import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas._testing import assert_frame_equal

from gaitmap.stride_segmentation.dtw_templates import DtwTemplate, create_dtw_template, BarthOriginalTemplate


class TestTemplateBaseClass:
    def test_template_provided(self):
        """Test very simple case where the template is directly stored in the class instance."""
        template = np.arange(10)

        instance = DtwTemplate(data=template)

        assert_array_equal(instance.data, template)

    def test_no_valid_info_provided(self):
        """Test that an error is raised, if neiter a filename nor a array is provided."""
        instance = DtwTemplate()

        with pytest.raises(AttributeError):
            _ = instance.data

    def test_use_columns_array(self):
        template = np.stack((np.arange(10), np.arange(10, 20))).T

        instance = DtwTemplate(data=template, use_cols=[1])

        assert_array_equal(instance.data, template[:, 1])

    def test_use_columns_dataframe(self):
        template = np.stack((np.arange(10), np.arange(10, 20))).T
        template = pd.DataFrame(template, columns=["col_1", "col_2"])

        instance = DtwTemplate(data=template, use_cols=["col_1"])

        assert_array_equal(instance.data, template[["col_1"]])

    def test_use_columns_wrong_dim(self):
        template = np.arange(10)

        instance = DtwTemplate(data=template, use_cols=[1])

        with pytest.raises(ValueError):
            _ = instance.data

    def test_load_from_file(self):
        instance = DtwTemplate(template_file_name="barth_original_template.csv")

        assert instance.data.shape == (200, 3)


class TestBartTemplate:
    def test_load(self):
        instance = DtwTemplate(template_file_name="barth_original_template.csv")

        barth_instance = BarthOriginalTemplate()

        assert_frame_equal(barth_instance.data, instance.data)
        assert barth_instance.sampling_rate_hz == 204.8
        assert barth_instance.scaling == 500.0


class TestCreateTemplate:
    def test_create_template_simple(self):
        template = np.arange(10)
        sampling_rate_hz = 100

        instance = create_dtw_template(template, sampling_rate_hz=sampling_rate_hz)

        assert_array_equal(instance.data, template)
        assert instance.sampling_rate_hz == sampling_rate_hz

    def test_create_template_use_col(self):
        template = np.stack((np.arange(10), np.arange(10, 20))).T
        template = pd.DataFrame(template, columns=["col_1", "col_2"])
        sampling_rate_hz = 100
        use_cols = ("col_1",)

        instance = create_dtw_template(template, sampling_rate_hz=sampling_rate_hz, use_cols=use_cols)

        assert_array_equal(instance.data, template[["col_1"]])
        assert instance.sampling_rate_hz == sampling_rate_hz
        assert_array_equal(instance.use_cols, use_cols)
