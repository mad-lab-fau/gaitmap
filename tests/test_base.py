"""This tests the BaseAlgorithm and fundamental functionality."""
from typing import Any, Dict, Tuple

import pytest

from gaitmap.base import BaseAlgorithm


def create_test_class(
    action_method_name, attributes=None, paras=None, other_paras=None, action_method=None
) -> BaseAlgorithm:
    attributes = attributes or {}
    paras = paras or {}
    other_paras = other_paras or {}

    class_dict = {"_action_method": action_method_name}
    user_set_paras = {
        **attributes,
        **paras,
        **other_paras,
    }
    if action_method:
        class_dict = {**class_dict, action_method_name: action_method}
    test_class = type("TestClass", (BaseAlgorithm,), class_dict)
    test_instance = test_class()
    for k, v in user_set_paras.items():
        setattr(test_instance, k, v)

    return test_instance


@pytest.fixture(
    params=[
        dict(action_method_name="test", attributes={}, paras={}, other_paras={}, action_method=None),
        dict(
            action_method_name="test",
            attributes={"attr1_": "test1", "attr2_": "test2"},
            paras={"para1": "test1", "para2": "test2"},
            other_paras={"other_para1": "other_test1", "other_para2": "other_test2"},
            action_method=lambda self=None: "test",
        ),
    ]
)
def example_test_class(request) -> Tuple[BaseAlgorithm, Dict[str, Any]]:
    test_class = create_test_class(**request.param)
    return test_class, request.param


def test_get_action_method(example_test_class):
    isntance, test_parameters = example_test_class

    assert instance._action_method == test_parameters["action_method_name"]
    if test_parameters["action_method"] is not None:
        assert instance._get_action_method()() == test_parameters["action_method"]()
    else:
        with pytest.raises(AttributeError):
            instance._get_action_method()


def test_get_attributes(example_test_class):
    instance, test_parameters = example_test_class

    assert instance.get_attributes() == test_parameters["attributes"]
