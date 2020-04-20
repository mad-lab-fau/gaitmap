"""Provide some example data to be used in simple tests.

The example data provides short sample data from foot mounted IMUs as well as calculated references from a camera based
Motion capture system for 2x20m walk test of a healthy subject.

The data is either taken from the local filesystem in case gaitlab was manually installed or you are asked to
download the data manually.
"""

from pathlib import Path

import pandas as pd

LOCAL_EXAMPLE_PATH = Path(__file__).parent.parent / "example_data/"
PC_EXAMPLE_PATH = Path.home() / ".gaitmap_data/"
GITLAB_FOLDER_PATH = (
    "https://mad-srv.informatik.uni-erlangen.de/newgaitpipeline/gaitmap/-/raw/master/example_data/{}?inline=false"
)


def _is_manual_installed() -> bool:
    return (LOCAL_EXAMPLE_PATH / "__init__.py").is_file()


def _get_data(filename: str) -> str:
    if _is_manual_installed():
        return LOCAL_EXAMPLE_PATH / filename
    if (PC_EXAMPLE_PATH / filename).is_file():
        return str(PC_EXAMPLE_PATH / filename)
    gitlab_path = GITLAB_FOLDER_PATH.format(filename)
    raise ValueError(
        "The gaitlab Python package does not contain the example data to save space. "
        'Please dowload the example folder manually from "{}" and place its content in the folder "{}". '
        'If the folder does not exist create it. Note the "." in front of the folder name.'.format(
            gitlab_path, PC_EXAMPLE_PATH
        )
    )


def get_healthy_example_imu_data():
    """Get example IMU data from a healthy subject doing a 2x20m gait test.

    The sampling rate is 204.8 Hz

    For expected results see:
        - :ref:`healthy_example_stride_borders`
    """
    test_data_path = _get_data("imu_sample.csv")
    data = pd.read_csv(test_data_path, header=[0, 1], index_col=0)

    # Get index in seconds
    data.index /= 204.8
    return data


def get_healthy_example_stride_borders():
    """Get hand labeled stride borders for :ref:`healthy_example_imu_data`.

    The stride borders are hand labeled at the gyr_ml minima before the toe-off.
    """
    test_data_path = _get_data("stride_borders_sample.csv")
    data = pd.read_csv(test_data_path, header=0)

    # Convert to dict with sensor name as key.
    # Sensor name here is derived from the foot. In the real pipeline that would be provided to the algo.
    data["sensor"] = data["foot"] + "_sensor"
    data = data.set_index("sensor")
    data = data.groupby(level=0)
    data = {k: v.reset_index(drop=True) for k, v in data}

    return data


def get_healthy_example_mocap_data():
    """Get 3D Mocap information of the foot synchronised with :ref:`healthy_example_imu_data`.

    The sampling rate is 100 Hz.

    The stride borders are hand labeled at the gyr_ml minima before the toe-off.
    """
    test_data_path = _get_data("mocap_sample.csv")
    data = pd.read_csv(test_data_path, header=[0, 1], index_col=0)

    # Get index in seconds
    data.index /= 100.0
    return data


def get_healthy_example_stride_events():
    """Get gait events extracted based on mocap data.

    The gait events are extracted based on the mocap data. but are converted to fit the indices of
    `healthy_example_imu_data`.

    However, because the values are extracted from another system the `min_vel` might not fit perfectly onto the imu
    data.

    This fixture returns a dictionary of stridelists, where the key is the sensor name.
    """
    test_data_path = _get_data("stride_events_sample.csv")
    data = pd.read_csv(test_data_path, header=0)

    # Convert to dict with sensor name as key.
    # Sensor name here is derived from the foot. In the real pipeline that would be provided to the algo.
    data["sensor"] = data["foot"] + "_sensor"
    data = data.set_index("sensor")
    data = data.groupby(level=0)
    data = {k: v.reset_index(drop=True) for k, v in data}
    return data


def get_healthy_example_orientation():
    """Get foot orientation calculated based on mocap data synchronised with :ref:`healthy_example_imu_data`.

    The sampling rate is 100 Hz.

    This data provides the orientation per stride in the default format.

    This fixture returns a dictionary, where the key is the sensor name.
    """
    test_data_path = _get_data("orientation_sample.csv")
    data = pd.read_csv(test_data_path, header=0, index_col=[0, 1, 2])

    # Convert to dict with sensor name as key.
    data = data.groupby(level=0)
    data = {k: v.reset_index(drop=True, level=0) for k, v in data}
    return data


def get_healthy_example_position():
    """Get foot position calculated based on mocap data synchronised with :ref:`healthy_example_imu_data`.

    The sampling rate is 100 Hz.

    This data provides the position per stride in the default format.
    The position is equivalent to the position of the heel marker.

    This fixture returns a dictionary, where the key is the sensor name.
    """
    test_data_path = _get_data("position_sample.csv")
    data = pd.read_csv(test_data_path, header=0, index_col=[0, 1, 2])

    # Convert to dict with sensor name as key.
    data = data.groupby(level=0)
    data = {k: v.reset_index(drop=True, level=0) for k, v in data}
    return data
