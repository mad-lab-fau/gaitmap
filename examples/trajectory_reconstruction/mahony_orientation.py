r"""
.. _trajectory_mahony:

Mahony and Madgwick Orientation Estimation
==========================================

This example compares :class:`~gaitmap.trajectory_reconstruction.MahonyAHRS` and
:class:`~gaitmap.trajectory_reconstruction.MadgwickAHRS` on the same single-sensor recording.
Both algorithms estimate orientation directly from IMU data, but they use different feedback mechanisms:
Mahony applies proportional-integral feedback, while Madgwick uses a gradient-descent correction step.

"""

# %%
# Getting input data
# ------------------
#
# We use a short section of the left foot IMU example data.
# The data is already aligned to the gaitmap sensor frame and contains accelerometer and gyroscope signals.
import matplotlib.pyplot as plt
import pandas as pd

from gaitmap.example_data import get_healthy_example_imu_data
from gaitmap.trajectory_reconstruction import MadgwickAHRS, MahonyAHRS

imu_data = get_healthy_example_imu_data()["left_sensor"].iloc[500:900]
sampling_frequency_hz = 204.8

# %%
# Configuring and running both algorithms
# ---------------------------------------
#
# `kp` controls how strongly the instantaneous accelerometer based correction is applied in Mahony.
# `ki` enables an accumulated bias correction term.
# Madgwick uses `beta` to control the correction strength.
mahony = MahonyAHRS(kp=0.8, ki=0.02)
madgwick = MadgwickAHRS(beta=0.2)

mahony = mahony.estimate(imu_data, sampling_rate_hz=sampling_frequency_hz)
madgwick = madgwick.estimate(imu_data, sampling_rate_hz=sampling_frequency_hz)

mahony_orientation = mahony.orientation_
madgwick_orientation = madgwick.orientation_
mahony_rotated_data = mahony.rotated_data_
madgwick_rotated_data = madgwick.rotated_data_

orientation_comparison = pd.concat(
    [mahony_orientation.add_prefix("mahony_"), madgwick_orientation.add_prefix("madgwick_")], axis=1
)
rotated_data_comparison = pd.concat(
    [mahony_rotated_data.add_prefix("mahony_"), madgwick_rotated_data.add_prefix("madgwick_")], axis=1
)

# %%
# Inspecting the results
# ----------------------
orientation_comparison.tail()

# %%
# Comparing the quaternion course can help to spot how strongly both filters react to the same movement.
orientation_comparison[["mahony_q_w", "madgwick_q_w"]].plot(figsize=(10, 4))
plt.xlabel("sample")
plt.ylabel("q_w [a.u.]")
plt.title("Mahony vs. Madgwick quaternion estimate")
plt.tight_layout()
plt.show()

# %%
# The rotated accelerometer data is often easier to interpret, because the sensor axes are aligned with the estimated
# global frame.
rotated_data_comparison[["mahony_acc_z", "madgwick_acc_z"]].plot(figsize=(10, 4))
plt.xlabel("time [s]")
plt.ylabel("acceleration [m/s^2]")
plt.title("Rotated vertical acceleration after orientation estimation")
plt.tight_layout()
plt.show()
