# This script will generate representations of the different coordinate definitions used within gaitmap
# The resulting plots are used within the coordinate system guide

import matplotlib.pyplot as plt

from gaitmap.example_data import get_healthy_example_imu_data
from gaitmap.preprocessing import sensor_alignment
from gaitmap.stride_segmentation import BarthDtw
from gaitmap.utils.consts import *
from gaitmap.utils.coordinate_conversion import convert_to_fbf

plt.close("all")

# Load gaitmap example dataset
example_dataset = get_healthy_example_imu_data()
sampling_rate_hz = 204.8

# Align dataset to gravity: resulting in sensor frame representation
dataset_sf = sensor_alignment.align_dataset_to_gravity(example_dataset, sampling_rate_hz)

# Convert dataset to body frame: resulting in body frame representation
dataset_bf = convert_to_fbf(dataset_sf, right=["right_sensor"], left=["left_sensor"])

# DTW Segmentation instance to get single strides
dtw = BarthDtw()  # create default BarthDtw instance
dtw = dtw.segment(dataset_bf, sampling_rate_hz=sampling_rate_hz)

# define doc colors
docs_red = np.array([148, 11, 41]) / 255.0
docs_green = np.array([0, 155, 121]) / 255.0
docs_blue = np.array([0, 181, 232]) / 255.0

colors = [docs_red, docs_green, docs_blue]

# helper to plot different coordinate frames
def plot_stride(data, column_names, sensor_id, stride_id, export_name):
    fig, axs = plt.subplots(2, figsize=(7, 7))
    start = dtw.stride_list_[sensor_id].iloc[stride_id].start
    end = dtw.stride_list_[sensor_id].iloc[stride_id].end
    axs[0].axhline(0, c="k", ls="--", lw=0.7)
    for i, col in enumerate(column_names[3:]):
        axs[0].plot(data[sensor_id][col].to_numpy()[start:end], label=col, color=colors[i])
    axs[0].set_title(col[:3] + " " + sensor_id)
    axs[0].legend(loc="upper right", bbox_to_anchor=(1.25, 1.0))
    axs[0].set_ylim([-600, 600])
    axs[0].set_ylabel("gyr [deg/s]")
    axs[0].set_xlabel("samples @204.8Hz")

    axs[1].axhline(-9.81, c="k", ls="--", lw=0.5)
    axs[1].axhline(+9.81, c="k", ls="--", lw=0.5)
    axs[1].axhline(0, c="k", ls="--", lw=0.7)
    for i, col in enumerate(column_names[:3]):
        axs[1].plot(data[sensor_id][col].to_numpy()[start:end], label=col, color=colors[i])
    axs[1].set_title(col[:3] + " " + sensor_id)
    axs[1].set_ylabel("acc [m/s^2]")
    axs[1].set_xlabel("samples @204.8Hz")
    axs[1].legend(loc="upper right", bbox_to_anchor=(1.25, 1.0))
    axs[1].set_ylim([-50, 50])
    plt.tight_layout()
    fig.savefig(export_name, bbox_inches="tight")
    fig.savefig(sensor_id + col[3:] + ".pdf", bbox_inches="tight")


#%%
# Plot "Stride-Template" in Sensor Frame
plot_stride(dataset_sf, SF_COLS, "left_sensor", 5, "left_sensor_sensor_frame_template.pdf")
plot_stride(dataset_sf, SF_COLS, "right_sensor", 18, "right_sensor_sensor_frame_template.pdf")

#%%
# Plot "Stride-Template" in Body Frame
plot_stride(dataset_bf, BF_COLS, "left_sensor", 5, "left_sensor_body_frame_template.pdf")
plot_stride(dataset_bf, BF_COLS, "right_sensor", 18, "right_sensor_body_frame_template.pdf")
