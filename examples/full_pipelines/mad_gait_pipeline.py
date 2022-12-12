r"""
MaD DiGait Pipeline
===================

This pipeline showcases the current gait analysis pipeline used by the MaD-Lab with all required steps:
Preprocessing -> Stride Segmentation -> Event Detection -> Trajectory Reconstruction -> Parameter Estimation

This should serve as a compact example that can be copied and pasted into new projects.
For more details on the individual steps have a look at the extended examples and the documentation of the main classes:

- :ref:`Preprocessing <example_preprocessing>`
- :ref:`Stride Segmentation (BarthDTW) <example_barth_stride_segmentation>`
- :ref:`Event Detection (RamppEventDetection) <example_rampp_event_detection>`
- :ref:`Trajectory Reconstruction (double Integration) <example_preprocessing>`
- :ref:`Temporal Parameters <example_temporal_parameters>` and :ref:`Spatial Parameters <example_spatial_parameters>`

"""
# %%
# Load example data
# -----------------
import numpy as np

from gaitmap.example_data import get_healthy_example_imu_data_not_rotated

np.random.seed(0)

example_dataset = get_healthy_example_imu_data_not_rotated()
sampling_rate_hz = 204.8

from gaitmap.preprocessing import sensor_alignment

# %%
# Preprocessing
# -------------
# Fix the alignment between the sensor coordinate system and the gaitmap coordinate system.
# This will be different for each sensor position and recording.
from gaitmap.utils.rotations import flip_dataset, rotation_from_angle

# rotate left_sensor first by -90 deg around the x-axis, followed by a -90 deg rotation around the z-axis
left_rot = rotation_from_angle(np.array([1, 0, 0]), np.deg2rad(-90)) * rotation_from_angle(
    np.array([0, 0, 1]), np.deg2rad(-90)
)

# rotate right_sensor first by +90 deg around the x-axis, followed by a +90 deg rotation around the z-axis
right_rot = rotation_from_angle(np.array([1, 0, 0]), np.deg2rad(90)) * rotation_from_angle(
    np.array([0, 0, 1]), np.deg2rad(90)
)

rotations = dict(left_sensor=left_rot, right_sensor=right_rot)
dataset_sf = flip_dataset(example_dataset, rotations)

# Align to Gravity
dataset_sf_aligned_to_gravity = sensor_alignment.align_dataset_to_gravity(dataset_sf, sampling_rate_hz)

from gaitmap.stride_segmentation import BarthDtw

# %%
# Stride Segmentation
# -------------------
# In this step the continuous datastream is segmented into individual strides.
# For longer datasets it might be required to first identify segments of walking to reduce the chance of
# false-positives.
from gaitmap.utils.coordinate_conversion import convert_to_fbf

dtw = BarthDtw()
# Convert data to foot-frame
bf_data = convert_to_fbf(dataset_sf_aligned_to_gravity, left_like="left_", right_like="right_")
dtw = dtw.segment(data=bf_data, sampling_rate_hz=sampling_rate_hz)

# %%
# Event detection
# ----------------
# For each identified stride, we now identify important stride events.
from gaitmap.event_detection import RamppEventDetection

ed = RamppEventDetection()
ed = ed.detect(data=bf_data, stride_list=dtw.stride_list_, sampling_rate_hz=sampling_rate_hz)

# %%
# Trajectory Reconstruction
# -------------------------
# Using the identified events the trajectory of each stride is reconstructed using double integration starting from the
# `min_vel` event of each stride.
from gaitmap.trajectory_reconstruction import StrideLevelTrajectory

trajectory = StrideLevelTrajectory()
trajectory = trajectory.estimate(
    data=dataset_sf, stride_event_list=ed.min_vel_event_list_, sampling_rate_hz=sampling_rate_hz
)

# %%
# Temporal Parameter Calculation
# ------------------------------
# Now we have all information to calculate relevant temporal parameters (like stride time)
from gaitmap.parameters import TemporalParameterCalculation

temporal_paras = TemporalParameterCalculation()
temporal_paras = temporal_paras.calculate(stride_event_list=ed.min_vel_event_list_, sampling_rate_hz=sampling_rate_hz)

# %%
# Spatial Parameter Calculation
# -----------------------------
# Like the temporal parameters, we can also calculate the spatial parameter.
from gaitmap.parameters import SpatialParameterCalculation

spatial_paras = SpatialParameterCalculation()
spatial_paras = spatial_paras.calculate(
    stride_event_list=ed.min_vel_event_list_,
    positions=trajectory.position_,
    orientations=trajectory.orientation_,
    sampling_rate_hz=sampling_rate_hz,
)

# %%
# Inspecting the Results
# ----------------------
# The class of each step allows you to inspect all results in detail.
# Here we will just print and plot the most important once.
# Note, that the plots below are for sure not the best way to represent results!
import matplotlib.pyplot as plt

print(
    "The following number of strides were identified and parameterized for each sensor: {}".format(
        {k: len(v) for k, v in ed.min_vel_event_list_.items()}
    )
)

# %%
for k, v in temporal_paras.parameters_pretty_.items():
    v.plot()
    plt.title("All temporal parameters of sensor {}".format(k))

# %%
for k, v in spatial_paras.parameters_pretty_.items():
    v[["stride length [m]", "gait velocity [m/s]", "arc length [m]"]].plot()
    plt.title("All spatial parameters of sensor {}".format(k))

# %%
for k, v in spatial_paras.parameters_pretty_.items():
    v.filter(like="angle").plot()
    plt.title("All angle parameters of sensor {}".format(k))
