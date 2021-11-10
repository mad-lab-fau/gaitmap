r"""
.. _example_sensor_alignment:

Sensor Alignment Simple
=======================

This example illustrates the sensor alignment pipeline, to make sure that the sensor coordinate frame is properly
aligned with the foot. This might be necessary e.g. in real-world datasets where participants attach and detach the
sensor frequently and possibly place the sensor in unintended orientations like upside down or 90/180deg rotated.
"""

import matplotlib.pyplot as plt

# %%
# Getting some example data
# -------------------------
#
# For this we take some example data that contains the regular walking movement during a 2x20m walk test of a healthy
# participant. The dataset is already calibrated and conforms to the gaitmap sensor frame axis convention.
# Furthermore, as this was a supervised recording, the alignment of the sensor to the shoe/ foot was manually aligned
# before the recording. The data contains synchronized data from two sensors - one from the right and one from the left
# foot.
from gaitmap.example_data import get_healthy_example_imu_data
from gaitmap.utils.consts import SF_ACC, SF_GYR

example_dataset = get_healthy_example_imu_data()
sampling_rate_hz = 204.8
# for simplicity we will only look at one foot in this example. However, all functions work the same way on both feet.
sensor = "right_sensor"

# %%
# Simulate some sensor misalignments
# ----------------------------------
# First we simulate some heavily misaligned data by applying some static rotations around each axis of the sensor frame.
# Afterwards we will apply some of the gaitmap preprocessing functions to automatically correct for all those
# misalignments. Therefore, we have to apply multiple steps to correct for different types of misalignment!
from scipy.spatial.transform import Rotation

from gaitmap.utils.rotations import rotate_dataset

# rotate the example data by some degree around each axis to simulate misalignment
z_axis_rotation = Rotation.from_euler("z", 70, degrees=True)
x_axis_rotation = Rotation.from_euler("x", 45, degrees=True)
y_axis_rotation = Rotation.from_euler("y", -250, degrees=True)

rotated_dataset = rotate_dataset(example_dataset, z_axis_rotation * x_axis_rotation * y_axis_rotation)

# %%
# Visualize the original and misaligned/ rotated dataset

_, axs = plt.subplots(2, 2, figsize=(13, 6))
axs[0, 0].plot(example_dataset[sensor].iloc[:5000][SF_ACC])
axs[0, 1].plot(rotated_dataset[sensor].iloc[:5000][SF_ACC])
for ax in axs[0]:
    ax.set_ylim([-150, 150])
    ax.grid("on")
axs[1, 0].plot(example_dataset[sensor].iloc[:5000][SF_GYR])
axs[1, 1].plot(rotated_dataset[sensor].iloc[:5000][SF_GYR])
for ax in axs[1]:
    ax.set_ylim([-850, 850])
    ax.grid("on")
axs[0, 0].set_title("Acceleration - original")
axs[1, 0].set_title("Gyroscope - original")
axs[0, 1].set_title("Acceleration - rotated")
axs[1, 1].set_title("Gyroscope - rotated")
plt.tight_layout()


# %%
# Sensor Alignment Pipeline
# -------------------------
# Now we apply all necessary steps for a full sensor to foot alignment procedure. This includes:
# * Gravity alignment
# * PCA alignment (as part of heading alignment)
# * Forward direction sign alignment (as part of heading alignment)

from gaitmap.preprocessing.sensor_alignment import ForwardDirectionSignAlignment, PcaAlignment, align_dataset_to_gravity

gravity_aligned_data = align_dataset_to_gravity(
    rotated_dataset, sampling_rate_hz=sampling_rate_hz, window_length_s=0.1, static_signal_th=15
)
pca_aligned_data = PcaAlignment().align(gravity_aligned_data).aligned_data_
forward_aligned_data = (
    ForwardDirectionSignAlignment().align(pca_aligned_data, sampling_rate_hz=sampling_rate_hz).aligned_data_
)

# %%
# Visualize the result of the gravity alignment

_, axs = plt.subplots(2, 3, figsize=(13, 6))
axs[0, 0].plot(example_dataset[sensor].iloc[:1000][SF_ACC])
axs[0, 1].plot(rotated_dataset[sensor].iloc[:1000][SF_ACC])
axs[0, 2].plot(forward_aligned_data[sensor].iloc[:1000][SF_ACC])
for ax in axs[0]:
    ax.set_ylim([-15, 15])
    ax.grid("on")

axs[1, 0].plot(example_dataset[sensor].iloc[1000:2000][SF_GYR])
axs[1, 1].plot(rotated_dataset[sensor].iloc[1000:2000][SF_GYR])
axs[1, 2].plot(forward_aligned_data[sensor].iloc[1000:2000][SF_GYR])
for ax in axs[1]:
    ax.set_ylim([-850, 850])
    ax.grid("on")

axs[0, 0].set_title("Acceleration - original")
axs[1, 0].set_title("Gyroscope - original")
axs[0, 1].set_title("Acceleration - rotated")
axs[1, 1].set_title("Gyroscope - rotated")
axs[0, 2].set_title("Acceleration - aligned")
axs[1, 2].set_title("Gyroscope - aligned")

plt.tight_layout()
