r"""
Region Level Trajectory reconstruction
======================================

This example shows how to calculate a IMU/foot trajectory over an entire gait sequence using
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

from gaitmap.example_data import get_healthy_example_imu_data
from gaitmap.trajectory_reconstruction import (
    SimpleGyroIntegration,
    ForwardBackwardIntegration,
    RegionLevelTrajectory,
    MadgwickAHRS,
)
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
# As we will perform the integration over a longer time period, we will use the Madgwick algorithm for the orientation,
# which is expected to perform better than just naive Gyro integration.
#
# For the spatial integration, we will use the forward-backward integration to compensate for drift.
# This is only possible, because our gait sequence starts and ends in a resting period.
#
# The initial orientation will automatically be estimated by the `RegionLevelTrajectory` class using the first
# n-samples.

ori_method = MadgwickAHRS(beta=0.02)
pos_method = ForwardBackwardIntegration()
trajectory = RegionLevelTrajectory(ori_method, pos_method)


# %%
# Calculate and inspect results
# -----------------------------
sampling_frequency_hz = 204.8
trajectory.estimate(data=imu_data, regions_of_interest=dummy_regions_list, sampling_rate_hz=sampling_frequency_hz)

# select the position of the first (and only) gait sequence
first_stride_position = trajectory.position_["left_sensor"].loc[0]

first_stride_position.plot()
plt.title("Left Foot Trajectory per axis")
plt.xlabel("sample")
plt.ylabel("position [m]")
plt.show()

# select the orientation of the first (and only) gait sequence
first_stride_orientation = trajectory.orientation_["left_sensor"].loc[0]

first_stride_orientation.plot()
plt.title("Left Foot Orientation per axis")
plt.xlabel("sample")
plt.ylabel("orientation [a.u.]")
plt.show()
