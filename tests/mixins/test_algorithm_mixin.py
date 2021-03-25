"""A mixin for all common tests that should be run on all algorithm classes."""
import inspect

import joblib
import pytest
from numpydoc.docscrape import NumpyDocString

from gaitmap.base import BaseType
from tests.conftest import compare_algo_objects


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

    def test_action_method_returns_self(self, after_action_instance):
        # call the action method a second time to test the output
        parameters = after_action_instance.get_other_params()
        results = getattr(after_action_instance, after_action_instance._action_method)(**parameters)

        assert id(results) == id(after_action_instance)

    def test_set_params_valid(self, after_action_instance):
        instance = after_action_instance.clone()
        valid_names = instance._get_param_names()
        values = list(range(len(valid_names)))
        instance.set_params(**dict(zip(valid_names, values)))

        for k, v in zip(valid_names, values):
            assert getattr(instance, k) == v, k

    def test_set_params_invalid(self, after_action_instance):
        instance = after_action_instance.clone()

        with pytest.raises(ValueError) as e:
            instance.set_params(an_invalid_name=1)

        assert "an_invalid_name" in str(e)
        assert self.algorithm_class.__name__ in str(e)

    def test_json_roundtrip(self, after_action_instance):
        instance = after_action_instance.clone()

        json_str = instance.to_json()

        instance_from_json = self.algorithm_class.from_json(json_str)

        compare_algo_objects(instance, instance_from_json)

    def test_hashing(self, after_action_instance):
        """This checks if caching with joblib will work as expected."""
        instance = after_action_instance.clone()

        assert joblib.hash(instance) == joblib.hash(instance.clone())
