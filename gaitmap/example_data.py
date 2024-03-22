"""Provide some example data to be used in simple tests.

The example data provides short sample data from foot mounted IMUs as well as calculated references from a camera based
Motion capture system for 2x20m walk test of a healthy subject.

The data is either taken from the local filesystem in case gaitmap was installed manually or it is automatically
downloaded to your cache folder (or to the path specified in the `GAITMAP_DATA_DIR` environment variable).
"""

from pathlib import Path

import pandas as pd
import pooch

from gaitmap import __version__

LOCAL_EXAMPLE_PATH = Path(__file__).parent.parent / "example_data/"
PC_EXAMPLE_PATH = Path.home() / ".gaitmap_data/"
GITHUB_FOLDER_PATH = "https://raw.githubusercontent.com/mad-lab-fau/gaitmap/{version}/example_data/"


BRIAN = pooch.create(
    # Use the default cache folder for the operating system
    path=pooch.os_cache("gaitmap"),
    # The remote data is on Github
    base_url=GITHUB_FOLDER_PATH,
    version=f"v{__version__}",
    version_dev="master",
    registry={
        "imu_sample.csv": "sha256:fdb91f0a1e58b1ac518a324d38c9177de6c4388137c1b1683e4a72460873bfd7",
        "imu_sample_healthy_stair_down.csv": "935442b1ec74ad69211ff3364e1d949c2811f1e24afa7f2f5322ce167976980e",
        "imu_sample_healthy_stair_up.csv": "91c66d4370fa2c152ab280df3ac2c607c71bf788f3a4bff7e690030aa6f5d5fe",
        "imu_sample_ms_left.csv": "5308a0833629a6ea76f3cdba3dca4ac185eabfd93e00b85f9cdaaa2a7c599b62",
        "imu_sample_ms_right.csv": "5349aeed45fbdb491c16f633b1f10a0d1c42b1bb2f56b429ce5fa9b9cd484376",
        "imu_sample_not_rotated.csv": "5136b5c7e086997d4a7423d9320a7abf2c8e467821920387157620959f427311",
        "mocap_sample.csv": "e27e89a75b38cb53520385eb252786c951a74a8f4a3d252322330ab672f6c00e",
        "orientation_sample.csv": "fe8f1c7fd12fccbbff916ab98f299b250f5c887634fbc2bb636fbcd02b114b0c",
        "position_sample.csv": "12cf9b837f51a01c1b5caf3e24d767307cf5d6619a372baf627481ff2c1e2703",
        "stride_borders_sample.csv": "6b93f875b7369bc9f6edd5842771dba79aef4e5922f7542786900852029ca914",
        "stride_events_sample.csv": "9fa47ac00ebe96fb6dc8447c49cf2ae1cf9559300101700a6875f094e4d5c274",
    },
    # The name of an environment variable that *can* overwrite the path
    env="GAITMAP_DATA_DIR",
)


def _is_manual_installed() -> bool:
    return (LOCAL_EXAMPLE_PATH / "__init__.py").is_file()


def _get_data(filename: str) -> str:
    if _is_manual_installed():
        return str(LOCAL_EXAMPLE_PATH / filename)

    # checks if file is already in local cache folder, otherwise downloads it from github; hashes are checked
    return BRIAN.fetch(filename)


def get_healthy_example_imu_data():
    """Get example IMU data from a healthy subject doing a 2x20m gait test.

    The sampling rate is 204.8 Hz
    """
    test_data_path = _get_data("imu_sample.csv")
    data = pd.read_csv(test_data_path, header=[0, 1], index_col=0)

    # Get index in seconds
    data.index /= 204.8
    return data


def get_ms_example_imu_data():
    """Get example IMU data from a MS subject performing a longer uninterrupted walking sequence.

    The sampling rate is 102.4 Hz and the data is not synchronised
    """
    data = {}
    for s in ["left", "right"]:
        test_data_path = _get_data(f"imu_sample_ms_{s}.csv")
        sensor_data = pd.read_csv(test_data_path, header=0, index_col=0)

        # Get index in seconds
        sensor_data.index /= 102.4
        data[s + "_sensor"] = sensor_data
    return data


def get_healthy_example_imu_data_not_rotated():
    """Get example IMU data from a healthy subject doing a 2x20m gait test.

    The sampling rate is 204.8 Hz
    """
    test_data_path = _get_data("imu_sample_not_rotated.csv")
    data = pd.read_csv(test_data_path, header=[0, 1], index_col=0)

    # Get index in seconds
    data.index /= 204.8
    return data


def get_healthy_example_stride_borders():
    """Get hand labeled stride borders for :func:`get_healthy_example_imu_data`.

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
    """Get 3D Mocap information of the foot synchronised with :func:`get_healthy_example_imu_data`.

    The sampling rate is 100 Hz.
    """
    test_data_path = _get_data("mocap_sample.csv")
    data = pd.read_csv(test_data_path, header=[0, 1], index_col=0)

    # Get index in seconds
    data.index /= 100.0
    return data


def get_healthy_example_stride_events():
    """Get gait events extracted based on mocap data.

    The gait events are extracted based on the mocap data. but are converted to fit the indices of
    :func:`get_healthy_example_imu_data`.

    However, because the values are extracted from another system the `min_vel` might not fit perfectly onto the imu
    data.

    This fixture returns a dictionary of stridelists, where the key is the sensor name.
    """
    test_data_path = _get_data("stride_events_sample.csv")
    data = pd.read_csv(test_data_path, header=0)

    # Convert to dict with sensor name as key.
    # Sensor name here is derived from the foot. In the real pipeline that would be provided to the algo.
    data["sensor"] = data["foot"] + "_sensor"
    data = data.set_index("sensor").drop("foot", axis=1)
    data = data.groupby(level=0)
    data = {k: v.reset_index(drop=True) for k, v in data}
    return data


def get_healthy_example_orientation():
    """Get foot orientation calculated based on mocap data synchronised with :func:`get_healthy_example_imu_data`.

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
    """Get foot position calculated based on mocap data synchronised with :func:`get_healthy_example_imu_data`.

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


def get_healthy_example_imu_data_stair_down():
    """Get example IMU data from a healthy subject walking down a single staircase.

    This data corresponds to the "stair_long_down_normal" of subject 11 in the MaD stair ambulation dataset

    The sampling rate is 204.8 Hz
    """
    test_data_path = _get_data("imu_sample_healthy_stair_down.csv")
    return pd.read_csv(test_data_path, header=[0, 1], index_col=0)


def get_healthy_example_imu_data_stair_up():
    """Get example IMU data from a healthy subject walking up a single staircase.

    This data corresponds to the "stair_long_up_normal" of subject 11 in the MaD stair ambulation dataset

    The sampling rate is 204.8 Hz
    """
    test_data_path = _get_data("imu_sample_healthy_stair_up.csv")
    return pd.read_csv(test_data_path, header=[0, 1], index_col=0)
