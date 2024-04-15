"""Extract example data from Fraunhofer dataset.

This script can be used to extract example data from the Frauenhofer dataset.
The dataset (as well as the required Python Packages) can be found on the Mad Gitlab:
https://mad-srv.informatik.uni-erlangen.de/MadLab/data/sensorpositoncomparison

Note: this script is meant to be independent of the gaitmap library and hence has a couple functions copied and pasted
here.
Note: This script will only work with older version of the libraries and dataset. Using the most update to date version
will likely not work. We will leave the script here as documentation, but it is unlikely to be reproducible.
"""

import warnings
from typing import Union

import numpy as np
import pandas as pd
from NilsPodLib.exceptions import CalibrationWarning, CorruptedPackageWarning, LegacyWarning, SynchronisationWarning
from scipy.spatial.transform import Rotation
from sensor_position_dataset_helper import (
    get_imu_test,
    get_mocap_test,
    get_session_df,
    get_subject_folder,
    get_subject_mocap_folder,
)

warnings.simplefilter("ignore", (LegacyWarning, CorruptedPackageWarning, CalibrationWarning, SynchronisationWarning))


def clip_to_closest_minima(data, points, radius):
    d = 2 * radius
    if len(data) - np.max(points) < radius:
        data = np.pad(data, (0, radius), constant_values=0)
    strides = np.lib.stride_tricks.as_strided(data, (len(data) - d, d), (data.strides[0], data.strides[0]))
    windows = strides[points.astype(int) - radius, :]
    return np.argmin(windows, axis=1) + points - radius


def rotation_from_angle(axis: np.ndarray, angle: Union[float, np.ndarray]) -> Rotation:
    """Create a quaternion based on a rotation axis and a angle.

    The output will be a normalized rotation

    Arguments:
        axis: normalized rotation axis
        angle: rotation angle in rad
    """
    angle = np.atleast_2d(angle)
    axis = np.atleast_2d(axis)
    return Rotation.from_rotvec(np.squeeze(axis * angle.T))


def rotate_sensor(data: pd.DataFrame, rotation: Rotation, inplace=False) -> pd.DataFrame:
    if inplace is False:
        data = data.copy()
    data[["gyr_" + i for i in list("xyz")]] = rotation.apply(data[["gyr_" + i for i in list("xyz")]].to_numpy())
    data[["acc_" + i for i in list("xyz")]] = rotation.apply(data[["acc_" + i for i in list("xyz")]].to_numpy())
    return data


def find_plane_from_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Get the normal vector of a plane defined by three points."""
    v1 = p2 - p1
    v2 = p3 - p1

    return normalize(np.cross(v1, v2))


def normalize(v: np.ndarray) -> np.ndarray:
    """Simply normalize a vector.

    If a 2D array is provided, each row is considered a vector, which is normalized independently.
    """
    v = np.array(v)
    ax = 0 if len(v.shape) == 1 else 1
    return (v.T / np.linalg.norm(v, axis=ax)).T


sampling_rate = 204.8
sampling_rate_mocap = 100
subject = "5047"
test = "normal_20"
sensor = "lateral"

full_dataset = get_session_df(subject)
test_df = get_imu_test(subject, test, session_df=full_dataset)
start = full_dataset.index.get_loc(test_df.index[0])
end = full_dataset.index.get_loc(test_df.index[-1])
test_df = test_df.reset_index(drop=True).drop("sync", axis=1)
test_df = test_df[["r_" + sensor, "l_" + sensor]]

test_mocap = get_mocap_test(subject, test)
test_mocap = test_mocap[["L_TOE", "R_TOE", "L_FCC", "R_FCC", "R_FM5", "L_FM5"]]
test_mocap.to_csv("./mocap_sample.csv")

test_borders = pd.read_csv(get_subject_folder(subject) / "manual_stride_border.csv", index_col=0)
test_borders = test_borders[(test_borders["start"] > start) & (test_borders["stop"] < end)]
test_borders -= start
# some refinement
# Flip the left foot as the Gyro is inverted for the lateral sensor
test_df_inverted = test_df.copy()
test_df_inverted["l_" + sensor] *= -1
test_borders = (
    test_borders.unstack()
    .groupby(level=[0, 1])
    .apply(
        lambda x: pd.Series(
            clip_to_closest_minima(test_df_inverted[x.name[1][0] + "_" + sensor]["gyr_z"].to_numpy(), x.to_numpy(), 20)
        )
    )
)
test_borders = test_borders.unstack(level=0).reset_index(level=-1, drop=True)
test_borders = test_borders.rename({"stop": "end"}, axis=1)
test_borders = test_borders.reset_index()
test_borders.index.name = "s_id"
test_borders = test_borders.reset_index()
test_borders["gsd_id"] = 1

test_borders.to_csv("./stride_borders_sample.csv", index=False)

# Rename columns and align with the expected orientation
left_rot = (
    rotation_from_angle(np.array([0, 0, 1]), np.deg2rad(90)) * rotation_from_angle(np.array([1, 0, 0]), np.deg2rad(90))
).inv()
right_rot = (
    rotation_from_angle(np.array([0, 0, 1]), np.deg2rad(-90))
    * rotation_from_angle(np.array([1, 0, 0]), np.deg2rad(-90))
).inv()
rotations = {"left_sensor": left_rot, "right_sensor": right_rot}
test_df = test_df.rename(columns={f"l_{sensor}": "left_sensor", f"r_{sensor}": "right_sensor"})
test_df.columns = test_df.columns.set_names(("sensor", "axis"))
test_df.sort_index(axis=1).to_csv("./imu_sample_not_rotated.csv")

test_df = (
    test_df.stack(level=0)
    .swaplevel()
    .groupby(level=0)
    .apply(lambda x: rotate_sensor(x, rotations[x.name]))
    .unstack(level=0)
    .swaplevel(axis=1)
)
test_df.columns = test_df.columns.set_names(("sensor", "axis"))
test_df.to_csv("./imu_sample.csv")

# Example events
test_events = test_borders = pd.read_csv(get_subject_mocap_folder(subject) / f"{test}_steps.csv", index_col=0)
test_events = test_events.rename(columns={"hs": "ic", "to": "tc", "ms": "min_vel"})
# convert to 204.8 Hz
test_events[["ic", "tc", "min_vel"]] *= 204.8 / 100
test_events[["ic", "tc", "min_vel"]] = test_events[["ic", "tc", "min_vel"]].round()
# Convert stride list to ms to ms
test_events["start"] = test_events["min_vel"]
test_events["end"] = test_events["start"].groupby(level=0).shift(-1)
test_events = test_events.drop("len", axis=1)
test_events["pre_ic"] = test_events["ic"]
test_events["ic"] = test_events["ic"].groupby(level=0).shift(-1)
test_events = test_events[~test_events["end"].isna()]
test_events["gsd_id"] = 1
test_events = test_events.reset_index()
test_events.index.name = "s_id"
test_events = test_events.reset_index()
test_events = test_events[["s_id", "foot", "start", "end", "ic", "tc", "min_vel", "pre_ic"]]

test_events.to_csv("./stride_events_sample.csv", index=False)

# Calculate orientation from mocap

# Back to 100 Hz
test_events[["start", "end"]] *= 100 / 204.8

test_orientation = {}
test_position = {}
for sensor, short in [("left_sensor", "L"), ("right_sensor", "R")]:
    normal_vectors = find_plane_from_points(
        test_mocap[f"{short}_FCC"], test_mocap[f"{short}_TOE"], test_mocap[f"{short}_FM5"]
    )
    forward_vector = normalize((test_mocap[f"{short}_FCC"] - test_mocap[f"{short}_TOE"]).to_numpy())
    sidewards = np.cross(normal_vectors, forward_vector, axis=1)
    rot_mat = np.hstack([forward_vector, sidewards, normal_vectors]).reshape((-1, 3, 3))
    ori = pd.DataFrame(Rotation.from_matrix(rot_mat).inv().as_quat(), columns=["q_x", "q_y", "q_z", "q_w"])
    ori_per_stride = {}
    pos_per_stride = {}
    for _, s in test_events[test_events["foot"] == sensor.split("_")[0]].iterrows():
        ori_per_stride[s["s_id"]] = ori.iloc[int(s["start"]) : int(s["end"])].reset_index(drop=True)
        pos = test_mocap[short + "_FCC"].iloc[int(s["start"]) : int(s["end"])].reset_index(drop=True)
        pos = pos - pos.iloc[0]  # Make it relative for each stride
        pos /= 1000  # from mm to m
        pos_per_stride[s["s_id"]] = pos
    ori_per_stride = pd.concat(ori_per_stride)
    ori_per_stride.index = ori_per_stride.index.rename(("s_id", "sample"))
    pos_per_stride = pd.concat(pos_per_stride)
    pos_per_stride.index = pos_per_stride.index.rename(("s_id", "sample"))
    pos_per_stride = pos_per_stride.add_prefix("pos_")

    test_orientation[sensor] = ori_per_stride
    test_position[sensor] = pos_per_stride

test_orientation = pd.concat(test_orientation)
test_position = pd.concat(test_position)
test_orientation.to_csv("./orientation_sample.csv")
test_position.to_csv("./position_sample.csv")

# Addition to fix the orientation bug #187

test_orientation = pd.read_csv("./orientation_sample.csv", index_col=[0, 1, 2])
fixed_ori = pd.DataFrame(Rotation.from_quat(test_orientation).inv().as_quat(), columns=["q_x", "q_y", "q_z", "q_w"])
fixed_ori.index = test_orientation.index
fixed_ori.to_csv("./orientation_sample.csv")
