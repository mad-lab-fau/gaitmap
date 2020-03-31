from importlib.resources import open_text

import pandas as pd
import pytest

from tests import example_data


@pytest.fixture()
def healthy_example_imu_data():
    """Example IMU data from a healthy subject doing a 2x20m gait test.

    The sampling rate is 204.8 Hz

    For expected results see:
        - :ref:`healthy_example_stride_borders`
    """
    with open_text(example_data, "imu_sample.csv") as test_data:
        data = pd.read_csv(test_data, header=[0, 1], index_col=0)

    # Get index in seconds
    data.index /= 204.8
    return data


@pytest.fixture()
def healthy_example_stride_borders():
    """Hand labeled stride borders for :ref:`healthy_example_imu_data`.

    The stride borders are hand labeled at the gyr_ml minima before the toe-off.
    """
    with open_text(example_data, "stride_borders_sample.csv") as test_data:
        data = pd.read_csv(test_data, header=0)

    return data


@pytest.fixture()
def healthy_example_mocap_data():
    """3D Mocap information of the foot synchronised with :ref:`healthy_example_imu_data`.

    The sampling rate is 100 Hz.

    The stride borders are hand labeled at the gyr_ml minima before the toe-off.
    """
    with open_text(example_data, "mocap_sample.csv") as test_data:
        data = pd.read_csv(test_data, header=[0, 1], index_col=0)

    # Get index in seconds
    data.index /= 100.0
    return data
