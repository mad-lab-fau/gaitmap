import random
from typing import Any

import numpy as np
import pandas as pd
import pytest
from tpcp import BaseTpcpObject

from gaitmap.example_data import (
    get_healthy_example_imu_data,
    get_healthy_example_imu_data_ic_stride,
    get_healthy_example_mocap_data,
    get_healthy_example_orientation,
    get_healthy_example_position,
    get_healthy_example_stride_borders,
    get_healthy_example_stride_borders_ic_stride,
    get_healthy_example_stride_events,
    get_ms_example_imu_data,
)


@pytest.fixture(autouse=True)
def reset_random_seed() -> None:
    np.random.seed(10)
    random.seed(10)


healthy_example_imu_data = pytest.fixture()(get_healthy_example_imu_data)
healthy_example_imu_data_ic_stride = pytest.fixture()(get_healthy_example_imu_data_ic_stride)
ms_example_imu_data = pytest.fixture()(get_ms_example_imu_data)
healthy_example_stride_borders = pytest.fixture()(get_healthy_example_stride_borders)
healthy_example_stride_borders_ic_stride = pytest.fixture()(get_healthy_example_stride_borders_ic_stride)
healthy_example_mocap_data = pytest.fixture()(get_healthy_example_mocap_data)
healthy_example_stride_events = pytest.fixture()(get_healthy_example_stride_events)
healthy_example_orientation = pytest.fixture()(get_healthy_example_orientation)
healthy_example_position = pytest.fixture()(get_healthy_example_position)


def _get_params_without_nested_class(instance: BaseTpcpObject) -> dict[str, Any]:
    return {k: v for k, v in instance.get_params().items() if not hasattr(v, "get_params")}


def group_matches_by_type(matches):
    """Select matched rows for public event/stride matching assertions."""
    if isinstance(matches, dict):
        return {name: group_matches_by_type(frame) for name, frame in matches.items()}
    grouped = matches.groupby("match_type")
    return {
        kind: grouped.get_group(kind) if kind in grouped.groups else pd.DataFrame(columns=matches.columns)
        for kind in ("tp", "fp", "fn")
    }
