r"""
Full top-to-bottom
=============

This example illustrates the whole top-to-bottom pipeline:
preprocessing -> DTW -> EventDetection -> TrajectoryReconstruction -> ParameterEstimation
"""

# Getting some example data
from gaitmap.example_data import get_healthy_example_imu_data_not_rotated

example_dataset = get_healthy_example_imu_data_not_rotated()
sampling_rate_hz = 204.8
example_dataset.sort_index(axis=1).head(1)

# Rename columns and align with the expected orientation
import numpy as np
from gaitmap.utils.rotations import rotation_from_angle, rotate_dataset

# rotate left_sensor first by -90 deg around the x-axis, followed by a -90 deg rotation around the z-axis
left_rot = rotation_from_angle(np.array([1, 0, 0]), np.deg2rad(-90)) * rotation_from_angle(
    np.array([0, 0, 1]), np.deg2rad(-90)
)

# rotate right_sensor first by +90 deg around the x-axis, followed by a +90 deg rotation around the z-axis
right_rot = rotation_from_angle(np.array([1, 0, 0]), np.deg2rad(90)) * rotation_from_angle(
    np.array([0, 0, 1]), np.deg2rad(90)
)

rotations = dict(left_sensor=left_rot, right_sensor=right_rot)

dataset_sf = rotate_dataset(example_dataset, rotations)

# %%
# Align to Gravity

import matplotlib.pyplot as plt
from gaitmap.preprocessing import sensor_alignment

dataset_sf_aligned_to_gravity = sensor_alignment.align_dataset_to_gravity(dataset_sf, sampling_rate_hz)

# %%
# DTW

np.random.seed(0)

from gaitmap.utils.coordinate_conversion import convert_to_fbf

bf_data = convert_to_fbf(example_dataset, left_like="left_", right_like="right_")
