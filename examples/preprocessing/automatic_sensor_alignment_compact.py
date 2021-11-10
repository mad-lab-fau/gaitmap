r"""
.. _example_automatic_sensor_alignment_simple:

Automatic sensor alignment
==========================

This example illustrates a minimal version of an automatic sensor alignment pipeline, to make sure that the sensor
coordinate frame is properly aligned with the foot, independent of the actual sensor attachment. This might be necessary
e.g. in real-world datasets where participants attach and detach the sensor frequently and possibly place the sensor in
unintended orientations like upside down or 90/180deg rotated. A more detailed example can be found here:
:ref:`example_automatic_sensor_alignment_detailed`
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
# Sensor alignment pipeline
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
# Visualize the result of the sensor alignment

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

# %%
# Performance considerations
# --------------------------
# During alignment multiple copies of the input data are created and the calculated orientations and positions stored
# on the result objects.
# This can lead to substantial RAM usage.
# Therefore, it might be a good idea to explicitly delete the intermediate data and used algorithm objects after the
# alignment.
del gravity_aligned_data
del pca_aligned_data

# %%
# Troubleshooting
# ---------------
# The alignment of the sensor to the foot might fail during windows which do not contain enough valid gait data! The
# PCA alignment and forward direction sign alignment are based on assumptions which are valid during gait (including
# stair ambulation) however will most certainly not hold for either static windows or non gait activities!
#
# As this pipeline applies the same, single coordinate transformation (all rotations of the individual steps could be
# combined to a single one) to all samples within the given sequence, it should be applied only to sequences where a
# constant misalignment is assumed. This assumption is usually valid for at least one walking bout, as a major change in
# the sensor attachment is unlikely to happen during gait. However, in between walking bouts participants might attach
# and detach sensor units (especially during continuous real-world recordings) and therefore, misalignment might change
# between bouts. Therefore, it is recommended to apply the alignment steps for each bout individually instead of
# processing whole real-world datasets at once.
#
# Although default values were chosen to work hopefully on most movement sequences containing gait, the sensor alignment
# pipeline contains a bunch of tunable hyperparameters which might need to be adapted for your special case.
#
# **Hyperparameter Tuning:**
#
# **Gravity Alignment:**
# Refer to :ref:`example_preprocessing`
#
# **PCA Alignment:**
#
# * Here only axis definitions can be adapted, however the default values are already chosen to conform to gaitmap
#   coordinate conventions
#
# **Forward Direction Sign Alignment:**
#
# * This function relies on a valid trajectory reconstruction, therefore Orientation and Position method hyperparameters
#   might require some tuning.
# * Second the `baseline_velocity_threshold` can be increased to exclude the influence of regions without movement
#   (aka where no forward velocity is present) and therefore, cannot add any information about the forward direction.
#   This might be necessary if the proportion of non-movement samples is much higher than movement samples in the
#   respective processed sequence.
