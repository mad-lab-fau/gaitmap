r"""
Full top-to-bottom
=============

This example illustrates the whole top-to-bottom pipeline:
preprocessing -> DTW -> EventDetection -> TrajectoryReconstruction -> ParameterEstimation
"""

# Getting raw and not-rotated example data
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

from gaitmap.preprocessing import sensor_alignment

dataset_sf_aligned_to_gravity = sensor_alignment.align_dataset_to_gravity(dataset_sf, sampling_rate_hz)

# %%
# DTW

np.random.seed(0)

from gaitmap.utils.coordinate_conversion import convert_to_fbf

bf_data = convert_to_fbf(dataset_sf_aligned_to_gravity, left_like="left_", right_like="right_")

from gaitmap.stride_segmentation import BarthDtw

dtw = BarthDtw()

# Apply the dtw to the data
dtw = dtw.segment(data=bf_data, sampling_rate_hz=sampling_rate_hz)

# %%
# Inspecting the results
# ----------------------
# The main output is the `stride_list_`, which contains the start and the end of all identified strides.
# As we passed a dataset with two sensors, the output will be a dictionary.
stride_list_left = dtw.stride_list_["left_sensor"]
print("{} strides were detected.".format(len(stride_list_left)))
stride_list_left.head()

# %%
# Applying the event detection

from gaitmap.event_detection import RamppEventDetection

ed = RamppEventDetection()
# apply the event detection to the data
ed = ed.detect(bf_data, sampling_rate_hz, dtw.stride_list_)

# printing some results
stride_events_left = ed.stride_events_["left_sensor"]
print("Gait events for {} strides were detected.".format(len(stride_events_left)))
stride_events_left.head()

# trajectory reconstruction
from gaitmap.trajectory_reconstruction import SimpleGyroIntegration, ForwardBackwardIntegration, StrideLevelTrajectory

ori_method = SimpleGyroIntegration()
pos_method = ForwardBackwardIntegration()
trajectory = StrideLevelTrajectory(ori_method, pos_method)

trajectory.estimate(dataset_sf, ed.stride_events_, sampling_rate_hz)

# %%
# Temporal parameter calculation

from gaitmap.parameters.temporal_parameters import TemporalParameterCalculation

p = TemporalParameterCalculation()
p = p.calculate(stride_event_list=ed.stride_events_, sampling_rate_hz=sampling_rate_hz)

# %%
# spatial parameter calculation

from gaitmap.parameters.spatial_parameters import SpatialParameterCalculation

stride_event_list = ed.stride_events_
positions = trajectory.position_
orientations = trajectory.orientation_

p = SpatialParameterCalculation()
p = p.calculate(
    stride_event_list=stride_event_list,
    positions=positions,
    orientations=orientations,
    sampling_rate_hz=sampling_rate_hz,
)
