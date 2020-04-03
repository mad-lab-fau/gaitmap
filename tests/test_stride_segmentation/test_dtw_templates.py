import pytest
from numpy.testing import assert_array_equal

from gaitmap.stride_segmentation.dtw_templates import DtwTemplate, create_dtw_template
import numpy as np


class TestTemplateBaseClass:
    def test_template_provided(self):
        """Test very simple case where the template is directly stored in the class instance."""
        template = np.arange(10)

        instance = DtwTemplate()
        instance._template = template

        assert_array_equal(instance.template, template)

    def test_no_valid_info_provided(self):
        """Test that an error is raised, if neiter a filename nor a array is provided."""
        instance = DtwTemplate()

        with pytest.raises(AttributeError):
            _ = instance.template


class TestCreateTemplate:
    def test_create_template_simple(self):
        template = np.arange(10)
        sampling_rate_hz = 100

        instance = create_dtw_template(template, sampling_rate_hz=sampling_rate_hz)

        assert_array_equal(instance.template, template)
        assert instance.sampling_rate_hz == sampling_rate_hz
