"""A mixin for all common tests that should be run on all algorithm classes."""
import inspect

from numpydoc.docscrape import NumpyDocString
import pytest

from gaitmap.base import BaseType


class TestAlgorithmMixin:
    algorithm_class = None
    __test__ = False

    @pytest.fixture()
    def after_action_instance(self) -> BaseType:
        pass

    def test_init(self):
        """Test that all init paras are passed through untouched."""
        field_names = self.algorithm_class._get_param_names()
        init_dict = {k: k for k in field_names}

        test_instance = self.algorithm_class(**init_dict)

        for k, v in init_dict.items():
            assert getattr(test_instance, k) == v, k

    def test_empty_init(self):
        """Test that the class has only optional kwargs."""
        self.algorithm_class()

    def test_all_parameters_documented(self):
        docs = NumpyDocString(inspect.getdoc(self.algorithm_class))

        documented_names = set(p.name for p in docs["Parameters"])
        actual_names = set(self.algorithm_class._get_param_names())

        assert documented_names == actual_names

    def test_all_attributes_documented(self, after_action_instance):
        if not after_action_instance:
            pytest.skip("The testclass did not implement the correct `after_action_instance` fixture.")
        docs = NumpyDocString(inspect.getdoc(self.algorithm_class))

        documented_names = set(p.name for p in docs["Attributes"])
        actual_names = set(after_action_instance.get_attributes().keys())

        assert documented_names == actual_names

    def test_all_other_parameters_documented(self, after_action_instance):
        if not after_action_instance:
            pytest.skip("The testclass did not implement the correct `after_action_instance` fixture.")
        docs = NumpyDocString(inspect.getdoc(self.algorithm_class))

        documented_names = set(p.name for p in docs["Other Parameters"])
        actual_names = set(after_action_instance.get_other_params().keys())

        assert documented_names == actual_names
