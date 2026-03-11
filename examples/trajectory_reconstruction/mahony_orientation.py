r"""
.. _trajectory_mahony:

Mahony Orientation Estimation
=============================

This example demonstrates how to apply :class:`~gaitmap.trajectory_reconstruction.MahonyAHRS` directly to a
single-sensor recording.
This low-level interface is useful when you want direct access to the full orientation estimate and to the
proportional-integral correction gains of the Mahony filter.

"""

# %%
# Getting input data
# ------------------
#
# We use a short section of the left foot IMU example data.
# The data is already aligned to the gaitmap sensor frame and contains accelerometer and gyroscope signals.
import matplotlib.pyplot as plt

from gaitmap.example_data import get_healthy_example_imu_data
from gaitmap.trajectory_reconstruction import MahonyAHRS

imu_data = get_healthy_example_imu_data()["left_sensor"].iloc[500:900]
sampling_frequency_hz = 204.8

# %%
# Configuring and running the algorithm
# -------------------------------------
#
# `kp` controls how strongly the instantaneous accelerometer based correction is applied.
# `ki` enables an accumulated bias correction term.
mahony = MahonyAHRS(kp=0.8, ki=0.02)
mahony = mahony.estimate(imu_data, sampling_rate_hz=sampling_frequency_hz)

orientation = mahony.orientation_
rotated_data = mahony.rotated_data_

# %%
# Inspecting the results
# ----------------------
orientation.tail()

# %%
# The rotated accelerometer data is often easier to interpret, because the sensor axes are aligned with the estimated
# global frame.
rotated_data[["acc_x", "acc_y", "acc_z"]].plot(figsize=(10, 4))
plt.xlabel("time [s]")
plt.ylabel("acceleration [m/s^2]")
plt.title("Rotated acceleration after Mahony orientation estimation")
plt.tight_layout()
plt.show()

# %%
# We can also inspect the quaternion time series directly.
orientation.plot(figsize=(10, 4))
plt.xlabel("sample")
plt.ylabel("quaternion component [a.u.]")
plt.title("Mahony orientation estimate")
plt.tight_layout()
plt.show()
