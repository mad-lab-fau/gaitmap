import random
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from pandas._testing import assert_frame_equal, assert_series_equal
from scipy.spatial.transform import Rotation
from tpcp import BaseTpcpObject

from gaitmap.example_data import (
    get_healthy_example_imu_data,
    get_healthy_example_mocap_data,
    get_healthy_example_orientation,
    get_healthy_example_position,
    get_healthy_example_stride_borders,
    get_healthy_example_stride_events,
    get_ms_example_imu_data,
)
from tests._regression_utils import PyTestSnapshotTest

try:
    from pomegranate import GeneralMixtureModel, State
except ImportError:
    GeneralMixtureModel = None
    State = None


@pytest.fixture(autouse=True)
def reset_random_seed():
    np.random.seed(10)
    random.seed(10)


@pytest.fixture()
def snapshot(request):
    with PyTestSnapshotTest(request) as snapshot_test:
        yield snapshot_test


def pytest_addoption(parser):
    group = parser.getgroup("snapshottest")
    group.addoption(
        "--snapshot-update", action="store_true", default=False, dest="snapshot_update", help="Update the snapshots."
    )


healthy_example_imu_data = pytest.fixture()(get_healthy_example_imu_data)
ms_example_imu_data = pytest.fixture()(get_ms_example_imu_data)
healthy_example_stride_borders = pytest.fixture()(get_healthy_example_stride_borders)
healthy_example_mocap_data = pytest.fixture()(get_healthy_example_mocap_data)
healthy_example_stride_events = pytest.fixture()(get_healthy_example_stride_events)
healthy_example_orientation = pytest.fixture()(get_healthy_example_orientation)
healthy_example_position = pytest.fixture()(get_healthy_example_position)


def _get_params_without_nested_class(instance: BaseTpcpObject) -> Dict[str, Any]:
    return {k: v for k, v in instance.get_params().items() if not hasattr(v, "get_params")}


def compare_algo_objects(a, b):
    parameters = _get_params_without_nested_class(a)
    b_parameters = _get_params_without_nested_class(b)

    assert set(parameters.keys()) == set(b_parameters.keys())

    for p, value in parameters.items():
        json_val = b_parameters[p]
        compare_val(value, json_val, p)


def compare_val(value, json_val, name):
    if isinstance(value, BaseTpcpObject):
        compare_algo_objects(value, json_val)
    elif isinstance(value, np.ndarray):
        assert_array_equal(value, json_val)
    elif isinstance(value, (tuple, list)):
        assert len(value) == len(json_val)
        for i, (v, j) in enumerate(zip(value, json_val)):
            compare_val(v, j, f"{name}_{i}")
    elif isinstance(value, Rotation):
        assert_array_equal(value.as_quat(), json_val.as_quat())
    elif isinstance(value, pd.DataFrame):
        assert_frame_equal(value, json_val, check_dtype=False)
    elif isinstance(value, pd.Series):
        assert_series_equal(value, json_val)
    elif State is not None and isinstance(value, State):
        assert value.name == json_val.name
        assert_almost_equal(value.weight, json_val.weight)
        if value.distribution is None:
            assert value.distribution == json_val.distribution
        else:
            if not isinstance(value.distribution, GeneralMixtureModel):
                raise ValueError("We only support comparing state with GMM distributions")
            for d1, d2 in zip(value.distribution.distributions, json_val.distribution.distributions):
                assert d1.name == d2.name
                for p1, p2 in zip(d1.parameters, d2.parameters):
                    assert_almost_equal(p1, p2)
            assert_almost_equal(value.distribution.weights, json_val.distribution.weights)
    else:
        assert value == json_val, name
