import pytest

from gaitmap.example_data import (
    get_healthy_example_imu_data,
    get_healthy_example_stride_borders,
    get_healthy_example_mocap_data,
    get_healthy_example_stride_events,
    get_healthy_example_orientation,
    get_healthy_example_position,
)
from tests._regression_utils import PyTestSnapshotTest


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
