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

data = get_healthy_example_imu_data()
sampling_rate_hz = 204.8
data.sort_index(axis=1).head(1)

# %%
# Align to Gravity
# ----------------
# To account for different sensor positions at the shoe and related different orientations of the sensor we will align
# our sensor data with gravity, to have a harmonised coordinate system definition. Therefore, we will use a
# static-moment-detection, to derive the absolute sensor orientation based on static accelerometer windows.
from gaitmap.preprocessing import sensor_alignment
from gaitmap.utils.consts import *

dataset_aligned_to_gravity = sensor_alignment.align_dataset_to_gravity(
    data, sampling_rate_hz
)

plt.figure()
plt.plot(data["left_sensor"][SF_ACC].to_numpy())
plt.figure()
plt.plot(dataset_aligned_to_gravity["left_sensor"][SF_ACC].to_numpy())
plt.show()

