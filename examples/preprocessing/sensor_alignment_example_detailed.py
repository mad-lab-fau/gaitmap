r"""
.. _example_sensor_alignment:

Sensor Alignment Detailed
=========================

This example illustrates the sensor alignment pipeline, to make sure that the sensor coordinate frame is properly
aligned with the foot. This might be necessary e.g. in real-world datasets where participants attach and detach the
sensor frequently and possibly place the sensor in unintended orientations like upside down or 90/180deg rotated.
"""

import matplotlib.pyplot as plt

# %%
# Getting some example data
# -------------------------
#
# For this, we take some example data that contains stair ascending gait on a single, straight staircase of a healthy
# participant. The sensors were attached to the instep position. The dataset is already calibrated and conforms to the
# gaitmap sensor frame axis convention. Furthermore, as this was a supervised recording, the alignment of the sensor
# to the shoe/ foot was manually aligned before the recording.
# The data contains synchronized data from two sensors - one from the right and one from the left foot.
from gaitmap.example_data import get_healthy_example_imu_data_stair_up
from gaitmap.utils.consts import SF_ACC, SF_GYR

example_dataset = get_healthy_example_imu_data_stair_up()
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

fig, axs = plt.subplots(2, 2, figsize=(13, 6))
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
fig.tight_layout()


# %%
# Align to Gravity
# ----------------
# Although our input data already conforms to the gaitmap sensor frame axis convention we still need to make
# sure that the z-axis is aligned with gravity (defined by [0,0,1]) as required by the gaitmap sensor-frame definition.
# This step will correct for misalignments around the x- and y-axis. Therefore, we will use a static-moment-detection,
# to derive the absolute sensor orientation based on static accelerometer windows and find the shortest rotation to
# gravity. The sensor coordinate system will be finally rotated, such that all static accelerometer windows will be
# close to `acc = [0.0, 0.0, 9.81]`.

from gaitmap.preprocessing import align_dataset_to_gravity

gravity_aligned_data = align_dataset_to_gravity(
    rotated_dataset, sampling_rate_hz=sampling_rate_hz, window_length_s=0.1, static_signal_th=15
)

# %%
# Visualize the result of the gravity alignment

fig, axs = plt.subplots(2, 3, figsize=(13, 6))
axs[0, 0].plot(example_dataset[sensor].iloc[:1000][SF_ACC])
axs[0, 1].plot(rotated_dataset[sensor].iloc[:1000][SF_ACC])
axs[0, 2].plot(gravity_aligned_data[sensor].iloc[:1000][SF_ACC])
for ax in axs[0]:
    ax.set_ylim([-15, 15])
    ax.grid("on")

axs[1, 0].plot(example_dataset[sensor].iloc[1000:2000][SF_GYR])
axs[1, 1].plot(rotated_dataset[sensor].iloc[1000:2000][SF_GYR])
axs[1, 2].plot(gravity_aligned_data[sensor].iloc[1000:2000][SF_GYR])
for ax in axs[1]:
    ax.set_ylim([-850, 850])
    ax.grid("on")
axs[0, 0].set_title("Acceleration - original")
axs[1, 0].set_title("Gyroscope - original")
axs[0, 1].set_title("Acceleration - rotated")
axs[1, 1].set_title("Gyroscope - rotated")
axs[0, 2].set_title("Acceleration - gravity aligned")
axs[1, 2].set_title("Gyroscope - gravity aligned")

fig.tight_layout()

# %%
# Heading Alignment
# -----------------
# Now we have successfully aligned the z-axis with gravity, we still see some obvious misalignment within the gyroscope
# data. This can be explained by the fact the static accelerometer data (used for alignment of the z-axis) does not
# provide any heading information. To compensate for a potential heading misalignment we will make use of the fact
# that the main movement component during walking happens within the sagittal plane, which corresponds to the flexion/
# "rolling" of the foot.
# To find the misalignment of the heading to the sagittal plane we will perform a Principle Component Analysis (PCA) of
# the gyroscope data in the x-y plane. We assume that the main component found by the PCA corresponds to the
# medio-lateral axis.
#
# However, the used PCA implementation will define the sign of this main component pointing towards the highest
# variance. Therefore, we found the sagittal plane but still miss the actual forward direction. Hence we
# still might require a final 180 deg flip around the z-axis. This will heavily depend on your input data. In some cases
# the sign of the PCA might already match the forward direction.

from gaitmap.preprocessing.sensor_alignment import PcaAlignment

pca_alignment = PcaAlignment(target_axis="y", pca_plane_axis=("gyr_x", "gyr_y"))
pca_alignment = pca_alignment.align(gravity_aligned_data)
pca_aligned_data = pca_alignment.aligned_data_

# %%
# Visualize the result of the pca/ heading alignment

_, axs = plt.subplots(2, 3, figsize=(13, 6))
axs[0, 0].plot(example_dataset[sensor].iloc[:1000][SF_ACC])
axs[0, 1].plot(gravity_aligned_data[sensor].iloc[:1000][SF_ACC])
axs[0, 2].plot(pca_aligned_data[sensor].iloc[:1000][SF_ACC])
for ax in axs[0]:
    ax.set_ylim([-15, 15])
    ax.grid("on")

axs[1, 0].plot(example_dataset[sensor].iloc[1000:2000][SF_GYR])
axs[1, 1].plot(gravity_aligned_data[sensor].iloc[1000:2000][SF_GYR])
axs[1, 2].plot(pca_aligned_data[sensor].iloc[1000:2000][SF_GYR])
for ax in axs[1]:
    ax.set_ylim([-850, 850])
    ax.grid("on")

axs[0, 0].set_title("Acceleration - original")
axs[1, 0].set_title("Gyroscope - original")
axs[0, 1].set_title("Acceleration - gravity aligned")
axs[1, 1].set_title("Gyroscope - gravity aligned")
axs[0, 2].set_title("Acceleration - PCA aligned")
axs[1, 2].set_title("Gyroscope - PCA aligned")

plt.tight_layout()

# %%
# Visualize the process of the PCA alignment

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

fig.suptitle("Birds Eye View", fontsize="x-large")

axs[0].scatter(gravity_aligned_data[sensor]["gyr_x"], gravity_aligned_data[sensor]["gyr_y"], marker=".", alpha=0.5)
axs[1].scatter(pca_aligned_data[sensor]["gyr_x"], pca_aligned_data[sensor]["gyr_y"], marker=".", alpha=0.5)

for ax in axs:
    ax.arrow(
        0.0,
        0.0,
        0,
        500,
        head_width=25,
        head_length=25,
        linewidth=2,
        color="b",
        length_includes_head=True,
        ls=":",
        zorder=5,
    )
    ax.arrow(
        0.0,
        0.0,
        500,
        0,
        head_width=25,
        head_length=25,
        linewidth=2,
        color="orange",
        length_includes_head=True,
        ls=":",
        zorder=5,
    )
    ax.text(25, 500, "World X", c="b")
    ax.text(500, -60, "World Y", c="orange")
    ax.set_ylim([-800, 800])
    ax.set_xlim([-800, 800])
    ax.axvline(0, c="k", ls="--", alpha=0.5, zorder=-1)
    ax.axhline(0, c="k", ls="--", alpha=0.5, zorder=-1)
    ax.set_xlabel("gyr-x")
    ax.set_ylabel("gyr-y")

axs[0].text(460, 200, "Sensor X", c="lime")
axs[0].text(200, -500, "Sensor Y", c="r")
axs[0].arrow(
    0.0,
    0.0,
    pca_alignment.pca_[sensor].components_[0][0] * 500,
    pca_alignment.pca_[sensor].components_[0][1] * 500,
    head_width=25,
    head_length=25,
    linewidth=2,
    color="lime",
    length_includes_head=True,
)
axs[0].arrow(
    0.0,
    0.0,
    pca_alignment.pca_[sensor].components_[1][0] * 500,
    pca_alignment.pca_[sensor].components_[1][1] * 500,
    head_width=25,
    head_length=25,
    linewidth=2,
    color="r",
    length_includes_head=True,
)
axs[1].arrow(0.0, 0.0, 0, 500, head_width=25, head_length=25, linewidth=2, color="lime", length_includes_head=True)
axs[1].arrow(0.0, 0.0, 500, 0, head_width=25, head_length=25, linewidth=2, color="r", length_includes_head=True)
axs[1].text(-220, 500, "Sensor X", c="lime")
axs[1].text(500, 40, "Sensor Y", c="r")

axs[0].set_title("Before Alignment")
axs[1].set_title("After PCA Alignment")

plt.tight_layout()

# %%
# Check the resulting rotation object

import numpy as np

pca_rotation = pca_alignment.rotation_[sensor]

# %%
# Lets look at the rotation angles in degree.
# We see that the PCA alignment method applys a pure heading correction, which is similar to our initial misalignment
# around the z-axis of 70 deg.
rot_angles = np.rad2deg(pca_rotation.as_euler("xyz"))
print("X-rot: %.1f deg, Y-rot: %.1f deg, Z-rot: %.1f deg" % (rot_angles[0], rot_angles[1], rot_angles[2]))


# %%
# Forward Direction Sign Alignment
# --------------------------------
# To find the sign of the forward direction we will perform an actual trajectory reconstruction and consider the sign of
# the sensor velocity in the foot posterior-anterior direction. As we do not yet have any information about strides
# available we will use zero velocity detectors for drift compensation and apply a piecewise-linear-drift-compensation.
# To ensure that the forward direction is always aligned with the sensor frame, the heading component will be ignored
# during the transformation of the sensor- to the world-frame during trajectory reconstruction.

from gaitmap.preprocessing.sensor_alignment import ForwardDirectionSignAlignment
from gaitmap.trajectory_reconstruction.orientation_methods import MadgwickAHRS
from gaitmap.trajectory_reconstruction.position_methods import PieceWiseLinearDedriftedIntegration
from gaitmap.utils.consts import GRAV_VEC
from gaitmap.zupt_detection import NormZuptDetector

fdsa = ForwardDirectionSignAlignment(
    forward_direction="x",
    rotation_axis="z",
    baseline_velocity_threshold=0.2,
    ori_method=MadgwickAHRS(beta=0.1),
    zupt_detector_orientation_init=NormZuptDetector(
        sensor="acc", window_length_s=0.15, inactive_signal_threshold=0.01, metric="variance"
    ),
    pos_method=PieceWiseLinearDedriftedIntegration(
        NormZuptDetector(sensor="gyr", window_length_s=0.15, inactive_signal_threshold=15.0, metric="mean"),
        level_assumption=False,
        gravity=GRAV_VEC,
    ),
)
fdsa = fdsa.align(pca_aligned_data, sampling_rate_hz=sampling_rate_hz)
forward_aligned_data = fdsa.aligned_data_

# %%
# Visualize the process of the forward direction alignment

fig, axs = plt.subplots(2, 3, figsize=(13, 6))
axs[0, 0].plot(example_dataset[sensor].iloc[:1000][SF_ACC])
axs[0, 1].plot(pca_aligned_data[sensor].iloc[:1000][SF_ACC])
axs[0, 2].plot(forward_aligned_data[sensor].iloc[:1000][SF_ACC])
for ax in axs[0]:
    ax.set_ylim([-15, 15])
    ax.grid("on")

axs[1, 0].plot(example_dataset[sensor].iloc[1000:2000][SF_GYR])
axs[1, 1].plot(pca_aligned_data[sensor].iloc[1000:2000][SF_GYR])
axs[1, 2].plot(forward_aligned_data[sensor].iloc[1000:2000][SF_GYR])
for ax in axs[1]:
    ax.set_ylim([-850, 850])
    ax.grid("on")

axs[0, 0].set_title("Acceleration - original")
axs[1, 0].set_title("Gyroscope - original")
axs[0, 1].set_title("Acceleration - PCA aligned")
axs[1, 1].set_title("Gyroscope - PCA aligned")
axs[0, 2].set_title("Acceleration - forward aligned")
axs[1, 2].set_title("Gyroscope - forward aligned")

fig.tight_layout()

# %%
# Check the resulting rotation object

fdsa_rotation = fdsa.rotation_[sensor]

# %%
# Lets look at the rotation angles in degree.
rot_angles = np.rad2deg(fdsa_rotation.as_euler("xyz"))
print("X-rot: %.1f deg, Y-rot: %.1f deg, Z-rot: %.1f deg" % (rot_angles[0], rot_angles[1], rot_angles[2]))

# %%
# Troubleshooting
# ---------------
# The alignment of the sensor to the foot might fail during windows which do not contain enough valid gait data! The
# PCA alignment and forward direction sign alignment are based on assumptions which are valid during gait (including
# stair ambulation) however will most certainly not hold for either static windows or non gait activities!
#
# Although default values were chosen to work hopefully on most movement sequences containing gait, the sensor alignment
# pipeline contains a bunch of tunable hyperparameters which might need to be adapted for your special case.
#
# **Hyperparameter Tuning:**
#
# **Gravity Alignment:**
# * First you could try to **increase** the **threshold**: The threshold refers to the metric calculated over the given
#   window on the norm of the gyroscope. So given the default metric "median" this means, a window will be considered
#   static if the median of the gyroscope norm is lower than the given threshold within the window length.
#
# * Second you could try to **lower** the **window length**: The shorter the window length, the higher the chance that
#   there is sequence of samples which will be below your set threshold.
#
# **PCA Alignment:**
# * Here only axis definitions can be adapted, however the default values are already chosen to conform to gaitmap
#   coordinate conventions
#
# **Forward Direction Sign Alignment:**
# * This function relies on a valid trajectory reconstruction, therefore Orientation and Position method hyperparameters
#   might require some tuning.
# * Second the `baseline_velocity_threshold` can be increased to exclude the influence of static samples (aka where no
#   movement/ velocity is present) and therefore, cannot add any information about the forward direction.
