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
# Preprocessing
# -------------

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


# Align to Gravity
from gaitmap.preprocessing import sensor_alignment

dataset_sf_aligned_to_gravity = sensor_alignment.align_dataset_to_gravity(dataset_sf, sampling_rate_hz)

# %%
# DTW
# ----------------------

np.random.seed(0)

from gaitmap.utils.coordinate_conversion import convert_to_fbf

bf_data = convert_to_fbf(dataset_sf_aligned_to_gravity, left_like="left_", right_like="right_")

from gaitmap.stride_segmentation import BarthDtw

dtw = BarthDtw()

# Apply the dtw to the data
dtw = dtw.segment(data=bf_data, sampling_rate_hz=sampling_rate_hz)

# %%
# Event detection
# ----------------------

from gaitmap.event_detection import RamppEventDetection

ed = RamppEventDetection()
# apply the event detection to the data
ed = ed.detect(bf_data, sampling_rate_hz, dtw.stride_list_)

# printing some results
stride_events_left = ed.stride_events_["left_sensor"]
print("Gait events for {} strides were detected.".format(len(stride_events_left)))
stride_events_left.head()

# %%
# Trajectory Reconstruction
# ----------------------

from gaitmap.trajectory_reconstruction import SimpleGyroIntegration, ForwardBackwardIntegration, StrideLevelTrajectory

ori_method = SimpleGyroIntegration()
pos_method = ForwardBackwardIntegration()
trajectory = StrideLevelTrajectory(ori_method, pos_method)

trajectory.estimate(dataset_sf, ed.stride_events_, sampling_rate_hz)

# %%
# Temporal parameter calculation
# ----------------------

from gaitmap.parameters.temporal_parameters import TemporalParameterCalculation

p = TemporalParameterCalculation()
p = p.calculate(stride_event_list=ed.stride_events_, sampling_rate_hz=sampling_rate_hz)

# %%
# spatial parameter calculation
# ----------------------

from gaitmap.parameters.spatial_parameters import SpatialParameterCalculation

p = SpatialParameterCalculation()
p = p.calculate(
    stride_event_list=ed.stride_events_,
    positions=trajectory.position_,
    orientations=trajectory.orientation_,
    sampling_rate_hz=sampling_rate_hz,
)
