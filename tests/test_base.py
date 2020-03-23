"""This tests the BaseAlgorithm and fundamental functionality."""
from inspect import signature, Parameter
from typing import Any, Dict, Tuple

import pytest

from gaitmap.base import BaseAlgorithm


def create_test_class(action_method_name, params=None, private_params=None, action_method=None, **_) -> BaseAlgorithm:
    params = params or {}
    private_params = private_params or {}

    class_dict = {"_action_method": action_method_name, "__init__": lambda x: None}
    user_set_params = {**params, **private_params}
    if action_method:
        class_dict = {**class_dict, action_method_name: action_method}
    test_class = type("TestClass", (BaseAlgorithm,), class_dict)

    # Set the signature to conform to the expected conventions
    sig = signature(test_class.__init__)
    sig = sig.replace(parameters=(Parameter(k, Parameter.KEYWORD_ONLY) for k in params.keys()))
    test_class.__init__.__signature__ = sig

    test_instance = test_class()
    for k, v in user_set_params.items():
        setattr(test_instance, k, v)

    return test_instance


@pytest.fixture(
    params=[
        dict(
            action_method_name="test", attributes={}, params={}, other_params={}, private_params={}, action_method=None
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

    assert instance._action_method == test_parameters["action_method_name"]
    if test_parameters["action_method"] is not None:
        assert instance._get_action_method()() == test_parameters["action_method"]()
    else:
        with pytest.raises(AttributeError):
            instance._get_action_method()


def test_get_attributes(example_test_class_after_action):
    instance, test_parameters = example_test_class_after_action

    assert instance.get_attributes() == test_parameters["attributes"]


def test_get_parameter(example_test_class_after_action):
    instance, test_parameters = example_test_class_after_action

    assert instance.get_params() == test_parameters["params"]


def test_get_other_parameter(example_test_class_after_action):
    instance, test_parameters = example_test_class_after_action

    assert instance.get_other_params() == test_parameters["other_params"]
