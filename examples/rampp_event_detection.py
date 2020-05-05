r"""
Rampp event detection
============================

This example illustrates how the gait event detection by the :class:`~gaitmap.event_detection.RamppEventDetection`
can be used to detect gait events within a list of strides and the corresponding IMU signal.
The used implementation is based on the work of Rampp et al. [1]_.

.. [1] Rampp, A., Barth, J., Schülein, S., Gaßmann, K. G., Klucken, J., & Eskofier, B. M. (2014). Inertial
   sensor-based stride parameter calculation from gait sequences in geriatric patients. IEEE transactions on
   biomedical engineering, 62(4), 1089-1097.. https://doi.org/10.1109/TBME.2014.2368211
"""

import matplotlib.pyplot as plt
import numpy as np

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
# Getting the example stride list
# --------------------------
#
# For this we take the ground truth stride list provided with the example data. For new data this stride list can be
# generated by running the algorithms provided in the :module:`~gaitmap.stride_segmentation` module.
from gaitmap.example_data import get_healthy_example_stride_borders

stride_list = get_healthy_example_stride_borders()
stride_list_left = stride_list["left_sensor"]

# %%
# Preparing the data
# ------------------
# The data is expected to be in the gaitmap BF to be able to use the same template for the left and the right foot.
# Therefore, we need to transform the our dataset into the body frame.
from gaitmap.utils.coordinate_conversion import convert_to_fbf

# We use the `..._like` parameters to identify the data of the left and the right foot based on the name of the sensor.
bf_data = convert_to_fbf(data, left_like="left_", right_like="right_")

# %%
# Applying the event detection
# ----------------
# First we need to initialize the Rampp event detection.
# In most cases it is sufficient to keep all parameters at default.
from gaitmap.event_detection import RamppEventDetection

ed = RamppEventDetection()
# apply the event detection to the data
ed.detect(bf_data, sampling_rate_hz, stride_list)

# %%
# Inspecting the results
# ----------------------
# The main output is the `stride_events_`, which contains the samples of initial contact (ic), terminal contact (tc),
# and minimal velocity (min_vel).
# Furthermore it contains start and end of each stride, which are aligned to the min_vel samples.
# The start sample of each stride corresponds to the min_vel sample of that stride and the end sample corresponds to the
# min_vel sample of the subsequent stride.
# Furthermore, the `stride_events_` list provides the pre_ic which is the ic event of the previous stride in the
# stride list.
# As we passed a dataset with two sensors, the output will be a dictionary.
stride_events_left = ed.stride_events_["left_sensor"]
print("Gait events for {} strides were detected.".format(len(stride_list_left)))
stride_events_left.head()

# %%
# To get a better understanding of the results, we can plot the data and the gait events.
# The top row shows the `gyr_ml` axis, the lower row the `acc_pa` axis along with the gait events with indicators as
# described in the plot legend.
# The vertical lines show the start and end of the strides that are overlapping with the min_vel samples.
#
# Only the second sequence of strides of the left foot are shown.

fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 5))
ax1.plot(bf_data.reset_index(drop=True)["left_sensor"][["gyr_ml"]])
ax2.plot(bf_data.reset_index(drop=True)["left_sensor"][["acc_pa"]])

ic_idx = ed.stride_events_["left_sensor"]["ic"].to_numpy().astype(int)
tc_idx = ed.stride_events_["left_sensor"]["tc"].to_numpy().astype(int)
min_vel_idx = ed.stride_events_["left_sensor"]["min_vel"].to_numpy().astype(int)

for ax, sensor in zip([ax1, ax2], ["gyr_ml", "acc_pa"]):
    for i, stride in ed.stride_events_["left_sensor"].iterrows():
        ax.axvline(stride["start"], color="g")
        ax.axvline(stride["end"], color="r")

    ax.scatter(
        ic_idx, bf_data["left_sensor"][sensor].to_numpy()[ic_idx], marker="*", s=100, color="r", zorder=3, label="ic",
    )

    ax.scatter(
        tc_idx, bf_data["left_sensor"][sensor].to_numpy()[tc_idx], marker="p", s=50, color="g", zorder=3, label="tc",
    )

    ax.scatter(
        min_vel_idx,
        bf_data["left_sensor"][sensor].to_numpy()[min_vel_idx],
        marker="s",
        s=50,
        color="y",
        zorder=3,
        label="min_vel",
    )

    ax.grid(True)

ax1.set_title("Rampp event detection result")
ax1.set_ylabel("gyr_ml (°/s)")
ax2.set_ylabel("acc_pa [m/s^2]")
ax1.set_xlim(3600, 7200)
# ax1.set_xlim(1150, 1850)
plt.legend(loc="best")

fig.tight_layout()
fig.show()


# %%
# To better understand the concept of ic and pre_ic, let's take a closer look at the data and zoom in a bit more. We
# can see now that every stride has a pre_ic and especially in case of the first stride of a sequence this pre_ic is
# not an ic for any stride.
# It only serves as a pre_ic for the subsequent stride.

fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 5))
ax1.plot(bf_data.reset_index(drop=True)["left_sensor"][["gyr_ml"]])
ax2.plot(bf_data.reset_index(drop=True)["left_sensor"][["acc_pa"]])

pre_ic_idx = ed.stride_events_["left_sensor"]["pre_ic"].to_numpy().astype(int)

for ax, sensor in zip([ax1, ax2], ["gyr_ml", "acc_pa"]):
    for i, stride in ed.stride_events_["left_sensor"].iterrows():
        ax.axvline(stride["start"], color="g")
        ax.axvline(stride["end"], color="r")

    ax.scatter(
        ic_idx, bf_data["left_sensor"][sensor].to_numpy()[ic_idx], marker="*", s=100, color="r", zorder=3, label="ic",
    )

    ax.scatter(
        pre_ic_idx,
        bf_data["left_sensor"][sensor].to_numpy()[pre_ic_idx],
        marker="d",
        s=50,
        color="k",
        zorder=3,
        label="pre_ic",
    )

    ax.scatter(
        tc_idx, bf_data["left_sensor"][sensor].to_numpy()[tc_idx], marker="p", s=50, color="g", zorder=3, label="tc",
    )

    ax.scatter(
        min_vel_idx,
        bf_data["left_sensor"][sensor].to_numpy()[min_vel_idx],
        marker="s",
        s=50,
        color="y",
        zorder=3,
        label="min_vel",
    )

    ax.grid(True)

ax1.set_title("Rampp event detection result")
ax1.set_ylabel("gyr_ml (°/s)")
ax2.set_ylabel("acc_pa [m/s^2]")
ax1.set_xlim(350, 720)
plt.legend(loc="best")

fig.tight_layout()
fig.show()
