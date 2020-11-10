r"""
Region Level Trajectory reconstruction
======================================

This example shows how to calculate a IMU/foot trajectory_full over an entire gait sequence using
:class:`~gaitmap.trajectory_reconstruction.RegionLevelTrajectory`.
If you need an introduction to trajectory reconstruction in general, have a look at:ref`this example
<trajectory_stride>`.
"""

# %%
# Getting input data
# ------------------
#
# For this example we need raw IMU data and a ROI list.
# We use the available gaitmap example data, which is already in the correct gaitmap coordinate system.
# Note, that the data starts and ends in a resting period, which is important for most integration methods to work
# properly.
#
# Because we only have a single gait sequence in the data, we will create a fake gait sequence list, that goes from
# the start to the end of the dataset.
import matplotlib.pyplot as plt
import pandas as pd

from gaitmap.example_data import get_healthy_example_imu_data, get_healthy_example_stride_events
from gaitmap.trajectory_reconstruction import RegionLevelTrajectory, RtsKalman
from gaitmap.utils.dataset_helper import get_multi_sensor_dataset_names

imu_data = get_healthy_example_imu_data()
dummy_regions_list = pd.DataFrame([[0, len(imu_data["left_sensor"])]], columns=["start", "end"]).rename_axis("gs_id")
dummy_regions_list = {k: dummy_regions_list for k in get_multi_sensor_dataset_names(imu_data)}
dummy_regions_list["left_sensor"]


# %%
# Selecting and Configuring Algorithms
# ------------------------------------
#
# Like for the stride level method we need to choose a orientation and a position algorithm.
# However, as we want to perform integration over a long time period, methods that can take advatage over the multiple
# regions of zero velocity (ZUPTs) to perform corrections are preferable.
# Therefore, the best choice for such region is a Kalman Filter.
# As this takes care of both position and orientation estimation in one go, we can pass it as a `trajectory_method`.
#
# The initial orientation will automatically be estimated by the `RegionLevelTrajectory` class using the first
# n-samples.
trajectory_method = RtsKalman()
trajectory_full = RegionLevelTrajectory(trajectory_method=trajectory_method)


# %%
# Calculate and inspect results
# -----------------------------
sampling_frequency_hz = 204.8
trajectory_full.estimate(data=imu_data, regions_of_interest=dummy_regions_list, sampling_rate_hz=sampling_frequency_hz)

# select the position of the first (and only) gait sequence
first_region_position = trajectory_full.position_["left_sensor"].loc[0]

first_region_position.plot()
plt.title("Left Foot Trajectory per axis")
plt.xlabel("sample")
plt.ylabel("position [m]")
plt.show()

# select the orientation of the first (and only) gait sequence
first_region_orientation = trajectory_full.orientation_["left_sensor"].loc[0]

first_region_orientation.plot()
plt.title("Left Foot Orientation per axis")
plt.xlabel("sample")
plt.ylabel("orientation [a.u.]")
plt.show()

# %%
# Calculate results per stride
# ----------------------------
# However, usually we are not interested in the full trajectory_full of a region, but the we want to know the trajectory_full of
# the strides within it.
# We can do that by first calculating the trajectory_full for the entire region and then cutting out the parts of each
# stride.
# This can be done in a single step using `estimate_intersect`, which takes an additional stride list as input.
# After using this method, the result contains the stride level trajectories.
# Note, that compared to the results `StrideLevelTrajectory`, the initial position and orientation of each stride is
# different and simply taken from the calculated trajectory_full.
stride_list = get_healthy_example_stride_events()

trajectory_per_stride = trajectory_full.clone()
trajectory_per_stride.estimate_intersect(
    data=imu_data,
    regions_of_interest=dummy_regions_list,
    stride_event_list=stride_list,
    sampling_rate_hz=sampling_frequency_hz,
)

# select the position of the first stride
first_region_position = trajectory_per_stride.position_["left_sensor"].loc[0]

first_region_position.plot()
plt.title("Left Foot Trajectory per axis")
plt.xlabel("sample")
plt.ylabel("position [m]")
plt.show()

# select the orientation of the first stride
first_region_orientation = trajectory_per_stride.orientation_["left_sensor"].loc[0]

first_region_orientation.plot()
plt.title("Left Foot Orientation per axis")
plt.xlabel("sample")
plt.ylabel("orientation [a.u.]")
plt.show()
