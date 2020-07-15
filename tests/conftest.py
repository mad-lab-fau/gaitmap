import random

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas._testing import assert_frame_equal, assert_series_equal
from scipy.spatial.transform import Rotation

from gaitmap.example_data import (
    get_healthy_example_imu_data,
    get_healthy_example_stride_borders,
    get_healthy_example_mocap_data,
    get_healthy_example_stride_events,
    get_healthy_example_orientation,
    get_healthy_example_position,
)
from tests._regression_utils import PyTestSnapshotTest


@pytest.fixture(autouse=True)
def reset_random_seed():
    np.random.seed(10)
    random.seed(10)


@pytest.fixture
def snapshot(request):
    with PyTestSnapshotTest(request) as snapshot_test:
        yield snapshot_test


def pytest_addoption(parser):
    group = parser.getgroup("snapshottest")
    group.addoption(
        "--snapshot-update", action="store_true", default=False, dest="snapshot_update", help="Update the snapshots."
    )


healthy_example_imu_data = pytest.fixture()(get_healthy_example_imu_data)
healthy_example_stride_borders = pytest.fixture()(get_healthy_example_stride_borders)
healthy_example_mocap_data = pytest.fixture()(get_healthy_example_mocap_data)
healthy_example_stride_events = pytest.fixture()(get_healthy_example_stride_events)
healthy_example_orientation = pytest.fixture()(get_healthy_example_orientation)
healthy_example_position = pytest.fixture()(get_healthy_example_position)


def compare_algo_objects(a, b):
    parameters = a._get_params_without_nested_class()
    b_parameters = b._get_params_without_nested_class()

    assert set(parameters.keys()) == set(b_parameters.keys())

    for p, value in parameters.items():
        json_val = b_parameters[p]
        if isinstance(value, np.ndarray):
            assert_array_equal(value, json_val)
        elif isinstance(value, (tuple, list)):
            assert list(value) == list(json_val)
        elif isinstance(value, Rotation):
            assert_array_equal(value.as_quat(), json_val.as_quat())
        elif isinstance(value, pd.DataFrame):
            assert_frame_equal(value, json_val)
        elif isinstance(value, pd.Series):
            assert_series_equal(value, json_val)
        else:
            assert value == json_val, p
