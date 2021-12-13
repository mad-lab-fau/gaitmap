"""This tests the BaseAlgorithm and fundamental functionality."""
from inspect import Parameter, signature
from typing import Any, Dict, Tuple

import pytest
from tests.conftest import _get_params_without_nested_class

from gaitmap.base import BaseAlgorithm
from tpcp import get_action_method, get_results, get_action_methods_names, get_action_params, is_action_applied


def _init_getter():
    def _fake_init(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    return _fake_init


def create_test_class(action_method_name, params=None, private_params=None, action_method=None, **_) -> BaseAlgorithm:
    params = params or {}
    private_params = private_params or {}

    class_dict = {"_action_methods": action_method_name, "__init__": _init_getter()}
    user_set_params = {**params, **private_params}
    if action_method:
        class_dict = {**class_dict, action_method_name: action_method}
    test_class = type("TestClass", (BaseAlgorithm,), class_dict)

    # Set the signature to conform to the expected conventions
    sig = signature(test_class.__init__)
    sig = sig.replace(parameters=(Parameter(k, Parameter.KEYWORD_ONLY) for k in params.keys()))
    test_class.__init__.__signature__ = sig

    test_instance = test_class(**user_set_params)

    return test_instance


@pytest.fixture(
    params=[
        dict(
            action_method_name="test",
            attributes={"attr1_": "test1"},
            params={},
            other_params={},
            private_params={},
            action_method=None,
        ),
        dict(
            action_method_name="test",
            attributes={"attr1_": "test1", "attr2_": "test2"},
            params={"para1": "test1", "para2": "test2"},
            other_params={"other_para1": "other_test1", "other_para2": "other_test2"},
            private_params={"_private": "private_test"},
            action_method=lambda self=None: "test",
        ),
    ]
)
def example_test_class_initialised(request) -> Tuple[BaseAlgorithm, Dict[str, Any]]:
    test_instance = create_test_class(**request.param)
    return test_instance, request.param


@pytest.fixture()
def example_test_class_after_action(example_test_class_initialised) -> Tuple[BaseAlgorithm, Dict[str, Any]]:
    test_instance, params = example_test_class_initialised
    action_params = {
        **params["attributes"],
        **params["other_params"],
    }
    for k, v in action_params.items():
        setattr(test_instance, k, v)
    return test_instance, params


def test_get_action_method(example_test_class_after_action):
    instance, test_parameters = example_test_class_after_action

    assert get_action_methods_names(instance)[0] == test_parameters["action_method_name"]
    if test_parameters["action_method"] is not None:
        assert get_action_method(instance)() == test_parameters["action_method"]()
    else:
        with pytest.raises(AttributeError):
            get_action_method(instance)


def test_get_attributes(example_test_class_after_action):
    instance, test_parameters = example_test_class_after_action

    assert get_results(instance) == test_parameters["attributes"]


def test_get_parameter(example_test_class_after_action):
    instance, test_parameters = example_test_class_after_action

    assert instance.get_params() == test_parameters["params"]


def test_get_other_parameter(example_test_class_after_action):
    instance, test_parameters = example_test_class_after_action

    assert get_action_params(instance) == test_parameters["other_params"]


def test_normal_wrong_attr_still_raises_attr_error(example_test_class_initialised):
    instance, test_parameters = example_test_class_initialised

    key = "not_existend_without_underscore"

    with pytest.raises(AttributeError) as e:
        getattr(instance, key)

    assert "result" not in str(e.value)
    assert key in str(e.value)
    assert get_action_methods_names(instance)[0] not in str(e.value)


@pytest.mark.parametrize("key", ["wrong_with_", "wrong_without"])
def test_attribute_helper_after_action_wrong(example_test_class_after_action, key):
    instance, test_parameters = example_test_class_after_action

    if not test_parameters["attributes"]:
        pytest.skip("Invalid fixture for this test")

    with pytest.raises(AttributeError) as e:
        getattr(instance, key)

    assert "result" not in str(e.value)
    assert key in str(e.value)
    assert get_action_methods_names(instance)[0] not in str(e.value)


def test_action_is_not_applied(example_test_class_initialised):
    instance, _ = example_test_class_initialised

    assert is_action_applied(instance) is False


def test_action_is_applied(example_test_class_after_action):
    instance, test_parameters = example_test_class_after_action

    if not test_parameters["attributes"]:
        pytest.skip("Invalid fixture for this test")

    assert is_action_applied(instance) is True


def test_nested_get_params():
    nested_instance = create_test_class("nested", params={"nested1": "n1", "nested2": "n2"})
    top_level_params = {"test1": "t1"}
    test_instance = create_test_class("test", params={**top_level_params, "nested_class": nested_instance})

    params = test_instance.get_params()

    assert isinstance(params["nested_class"], nested_instance.__class__)

    for k, v in nested_instance.get_params().items():
        assert params["nested_class__" + k] == v

    for k, v in top_level_params.items():
        assert params[k] == v


def test_nested_set_params():
    nested_instance = create_test_class("nested", params={"nested1": "n1", "nested2": "n2"})
    top_level_params = {"test1": "t1"}
    test_instance = create_test_class("test", params={**top_level_params, "nested_class": nested_instance})
    new_params_top_level = {"test1": "new_t1"}
    new_params_nested = {"nested2": "new_n2"}
    test_instance.set_params(**new_params_top_level, **{"nested_class__" + k: v for k, v in new_params_nested.items()})

    params = test_instance.get_params()
    params_nested = nested_instance.get_params()

    for k, v in new_params_top_level.items():
        assert params[k] == v
    for k, v in new_params_nested.items():
        assert params["nested_class__" + k] == v
        assert params_nested[k] == v


def test_nested_clone():
    nested_instance = create_test_class("nested", params={"nested1": "n1", "nested2": "n2"})
    top_level_params = {"test1": "t1"}
    test_instance = create_test_class("test", params={**top_level_params, "nested_class": nested_instance})

    cloned_instance = test_instance.clone()

    # Check that the ids are different
    assert test_instance is not cloned_instance
    assert test_instance.nested_class is not cloned_instance.nested_class

    params = _get_params_without_nested_class(test_instance)
    cloned_params = _get_params_without_nested_class(cloned_instance)

    for k, v in params.items():
        assert cloned_params[k] == v
