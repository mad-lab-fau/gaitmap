r"""
.. _roi_stride_segmentation:

Stride segmentation with Regions of Interest
============================================

In long datasets it might be desirable to predefine regions of interest in which strides should be segmented.
This can massively speed up the computation or focus the analysis on a specific part of the data.

Most of the time such regions of interest would be defined using some sort of gait detection algorithms (e.g.
:mod:`~gaitmap.gait_detection.UllrichGaitSequenceDetection`).
In this example we will create an example output of such an algorithms and will use it to limit the search region for
`BarthDtw` stride segmentation algorithms.
However, all other stride segmentation algorithms should support this functionality as well.

Note, that for smaller datasets, the algorithms might actually be slower when using multiple regions of interest,
compared to analysing the entire dataset.
This is, because the method relies on a slow Python loop to process the individual regions of interest.
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
# We will further convert the data into the body frame, as this is required by the `BarthDTW` method to perform the
# stride segmentation.
from gaitmap.example_data import get_healthy_example_imu_data
from gaitmap.utils.coordinate_conversion import convert_to_fbf

data = get_healthy_example_imu_data()
sampling_rate_hz = 204.8
bf_data = convert_to_fbf(data, left_like="left_", right_like="right_")


data.sort_index(axis=1).head(1)

# %%
# Defining regions of interest
# ----------------------------
#
# We will define two regions of interest (one at the beginning and one at the end of the dataset).
# We expect the algorithm to only find strides in these regions.
import pandas as pd

roi = pd.DataFrame([[0, 2000], [5000, 7500]], columns=["start", "end"])
roi.index.name = "roi_id"
roi

# %%
# Setting up the Stride Segmentation
# ----------------------------------
# We will use BarthDTW with the default settings.
from gaitmap.stride_segmentation import BarthDtw

dtw = BarthDtw()

# %%
# Applying the DTW
# ----------------
# We apply the DTW in the same way as usual, but use the optional parameter `regions_of_interest` to define the
# search regions
dtw = dtw.segment(data=bf_data, sampling_rate_hz=sampling_rate_hz, regions_of_interest=roi)

# %%
# Inspecting the results
# ----------------------
# Regions of interest effect the outputs in multiple ways.
# First, the stride list contains an additional column called `roi_id` (or `gs_id`, depending on the input).
# It indicates to which region of interest the specific stride belongs
stride_list_left = dtw.stride_list_["left_sensor"]
print("{} strides were detected.".format(len(stride_list_left)))
stride_list_left

# %%
# Further, if we try to plot the results, we will see that we now have multiple cost matrices and cost functions per
# sensor.
# This is simply because the DTW was applied separately to the two regions.
# `dtw.acc_cost_mat_[<sensor name>]` and `dtw.cost_function_[<sensor name>]` are dictionaries of the form `{<roi_id> :
# ...}`.
# We can loop over them to plot its content, as shown below.
import numpy as np

sensor = "left_sensor"
fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(10, 5))
dtw.data[sensor]["gyr_ml"].reset_index(drop=True).plot(ax=axs[0])
axs[0].set_ylabel("gyro [deg/s]")
cost_funcs = dtw.cost_function_[sensor]
cost_mat = dtw.acc_cost_mat_[sensor]
full_cost_matrix = np.full((len(dtw.template.data), len(dtw.data)), np.nan)
for roi, (start, end) in dtw.regions_of_interest[["start", "end"]].iterrows():
    axs[1].plot(np.arange(start, end), cost_funcs[roi], c="C0")
    full_cost_matrix[:, start:end] = cost_mat[roi]
axs[1].set_ylabel("dtw cost [a.u.]")
axs[1].axhline(dtw.max_cost, color="k", linestyle="--")
axs[2].imshow(full_cost_matrix, aspect="auto")
axs[2].set_ylabel("template position [#]")
for p in dtw.paths_[sensor]:
    axs[2].plot(p.T[1], p.T[0])
for s in dtw.matches_start_end_original_[sensor]:
    axs[1].axvspan(*s, alpha=0.3, color="g")
for _, s in dtw.stride_list_[sensor][["start", "end"]].iterrows():
    axs[0].axvspan(*s, alpha=0.3, color="g")
axs[0].set_xlabel("time [#]")
fig.tight_layout()
fig.show()

# sphinx_gallery_thumbnail_number = 1
