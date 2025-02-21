r"""
Region Level Trajectory with Magnetometer
=========================================

While most algorithms in gaitmap are designed to only work with accelerometer and gyroscope data, we have implemented
the magnetometer version of the Madgwick algorithm (`use_magnetometer=True` in `MadgwickAHRS`).
This can be used as standalone orientation method or as part of the `MadgwickRtsKalman` trajectory method.

Below, we show an example recording (provided by the Fraunhofer IIS) of an X-Sense sensor attached at the foot.
The person walked L-shaped trajectory repeatedly.

In this example we will compare the results of the Kalman Filter with and without the use of the magnetometer data.

To simplify the example, we have already pre-aligned the data to the gaitmap coordinate system and ensured that the
data is in the correct format.


"""

# %%
# Getting input data
# ------------------
import matplotlib.pyplot as plt

from gaitmap.example_data import get_magnetometer_l_path_data
from gaitmap.trajectory_reconstruction import MadgwickRtsKalman, RtsKalman

imu_data = get_magnetometer_l_path_data().iloc[:100000]
sampling_frequency_hz = 400

# %%
# Selecting and Configuring Algorithms
# ------------------------------------
mad_with_mag = MadgwickRtsKalman(use_magnetometer=True, madgwick_beta=0.105, velocity_error_variance=0.01)
mad_without_mag = MadgwickRtsKalman(use_magnetometer=False)
simple = RtsKalman()

# %%
# Calculate and inspect results
# -----------------------------
mad_with_mag.estimate(data=imu_data, sampling_rate_hz=sampling_frequency_hz)
mad_without_mag.estimate(data=imu_data, sampling_rate_hz=sampling_frequency_hz)
simple.estimate(data=imu_data, sampling_rate_hz=sampling_frequency_hz)

with_mag_traj = mad_with_mag.position_
without_mag_traj = mad_without_mag.position_
simple_traj = simple.position_

# %%
# Plotting
# --------
# We are plotting the x-y plane to see if the magnetometer has a relevant influence on the trajectory.
# In particular, we assume that the heading of the foot is more accurate when using the magnetometer and does
# not drift as much as without it.
plt.figure(figsize=(12, 6))
plt.plot(simple_traj["pos_x"], simple_traj["pos_y"], label="Simple Kalman")
plt.plot(without_mag_traj["pos_x"], without_mag_traj["pos_y"], label="Madgwick Kalman without Magnetometer")
plt.plot(with_mag_traj["pos_x"], with_mag_traj["pos_y"], label="Madgwick Kalman with Magnetometer")

plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.axis("equal")

plt.legend()
plt.show()

# %%
# Conclusion
# ----------
# We can see that with the magnetometer the heading is somewhat stable.
# While there is still a positional drift between the repetitions, all the walking path align with the same axis.
# For the other two methods, we see less positional drift, but the heading drifts significantly.
#
# The positional drift around x=0 is likely caused by large metallic objects in the room, which distort the magnetic
# field.
# Both other methods don't show comparable artifacts in that region.
#
# In all three cases, tuning the parameters of the Kalman filter could help to further reduce drift.
# However, without the magnetometer, there is little chance to get a stable heading.
