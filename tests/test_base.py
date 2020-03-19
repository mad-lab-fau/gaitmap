"""This tests the BaseAlgorithm and fundamental functionality."""
from typing import Any, Dict, Tuple, Type

import pytest

from gaitmap.base import BaseAlgorithm


def create_test_class(action_method_name, attributes=None, paras=None, other_paras=None, action_method=None):
    attributes = attributes or {}
    paras = paras or {}
    other_paras = other_paras or {}

    class_dict = {**attributes, **paras, **other_paras, "_action_method": action_method_name}
    if action_method:
        class_dict = {**class_dict, action_method_name: action_method}
    return type("TestClass", (BaseAlgorithm,), class_dict)


@pytest.fixture(
    params=[
        dict(action_method_name="test", attributes=None, paras=None, other_paras=None, action_method=None),
        dict(
            action_method_name="test",
            attributes={"_attr1": "test1", "_attr2": "test2"},
            paras={"para1": "test1", "para2": "test2"},
            other_paras={"other_para1": "other_test1", "other_para2": "other_test2"},
            action_method=lambda self=None: "test",
        ),
    ]
)
def example_test_class(request) -> Tuple[Type[BaseAlgorithm], Dict[str, Any]]:
    test_class = create_test_class(**request.param)
    return test_class, request.param


def test_get_action_method(example_test_class):
    test_class, test_parameters = example_test_class
    instance = test_class()

    assert instance._action_method == test_parameters["action_method_name"]
    if test_parameters["action_method"] is not None:
        assert instance._get_action_method()() == test_parameters["action_method"]()
    else:
        with pytest.raises(AttributeError):
            instance._get_action_method()
