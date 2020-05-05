r"""
Preprocessing
=============

This example illustrates the preprocessing pipeline, to make sure that coordinate system conventions are correct for the
use of all gaitmap functions.
"""

import matplotlib.pyplot as plt
import numpy

numpy.random.seed(0)

# %%
# Getting some example data
# --------------------------
#
# For this we take some example data that contains the regular walking movement during a 2x20m walk test of a healthy
# subject. The IMU signals are already rotated so that they align with the gaitmap SF coordinate system.
# The data contains information from two sensors - one from the right and one from the left foot.
from gaitmap.example_data import get_healthy_example_imu_data

example_dataset = get_healthy_example_imu_data()
sampling_rate_hz = 204.8
example_dataset.sort_index(axis=1).head(1)

# %%
# Calibration to physical units
# -----------------------------
# All gaitmap algorithms expect the sensor data to be properly calibrated to physical units. For accelerometer
# units should be m/s^2 and for gyroscope deg/s.

# %%
# Align to Gravity
# ----------------
# To account for different sensor positions at the shoe and related different orientations of the sensor we will align
# our sensor data with gravity, to have a harmonised coordinate system definition. Therefore, we will use a
# static-moment-detection, to derive the absolute sensor orientation based on static accelerometer windows.
# The sensor coordinate system will be rotated, such that all static accelerometer windows will be close to
# acc = [0.0, 0.0, 9.81]
from gaitmap.preprocessing import sensor_alignment
from gaitmap.utils.consts import *

dataset_aligned_to_gravity = sensor_alignment.align_dataset_to_gravity(example_dataset, sampling_rate_hz)

# %%
# Vizualize the result

fig, (ax0, ax1) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 5))
ax0.set_title("Original Data")
ax0.set_ylabel("acc [m/s^2]")
ax0.plot(example_dataset["left_sensor"][SF_ACC].to_numpy()[:700, :])
ax0.axhline(0, c="k", ls="--", lw=0.5)
ax0.axhline(9.81, c="k", ls="--", lw=0.5)

ax1.set_title("Aligned to Gravity")
ax1.plot(dataset_aligned_to_gravity["left_sensor"][SF_ACC].to_numpy()[:700, :])
ax1.axhline(0, c="k", ls="--", lw=0.5)
ax1.axhline(9.81, c="k", ls="--", lw=0.5)
ax1.set_ylim([-50, 50])
plt.show()

# %%
# Troubleshooting
# ---------------
# The align_dataset_to_gravity function might fail to find static windows within your input signal and therefore also
# fail to align your signal. In this case you might need to tweak the default parameters a bit:
# * First you could try to increase the threshold: The threshold refers to the metric calculated over the given window
# on the norm of the gyroscope. So given the metric "mean" this means, a window will be considered static if the mean
# of the gyrocsope norm is lower than the given threshold within the window length.
# * Second you could try to lower the window length: The shorter the window length, the higher the chance that there is
# seqeunce of samples which will be below your set threshold.