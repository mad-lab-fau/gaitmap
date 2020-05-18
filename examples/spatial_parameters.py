r"""
Spatial parameters calculation
==============================

This example illustrates illustrates how spatial parameters can be calculated for each stride  by
the :class:`~gaitmap.parameters.SpatialParameterCalculation`.
The used implementation is based on the work of Kanzler et al [1]_ and Rampp et al [2]_.

.. [1] Kanzler, C. M., Barth, J., Rampp, A., Schlarb, H., Rott, F., Klucken, J., Eskofier, B. M. (2015, August).
       Inertial sensor based and shoe size independent gait analysis including heel and toe clearance estimation.
       In 2015 37th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)
       (pp. 5424-5427). IEEE.
.. [2] Rampp, A., Barth, J., Schülein, S., Gaßmann, K. G., Klucken, J., & Eskofier, B. M. (2014).
       Inertial sensor-based stride parameter calculation from gait sequences in geriatric patients.
       IEEE transactions on biomedical engineering, 62(4), 1089-1097.
"""

# %%
# Getting input data
# ------------------
#
# For this we need stride event list obtained from event detection method, position and orientation list
# obtained from trajectory reconstruction.
from gaitmap.example_data import (
    get_healthy_example_stride_events,
    get_healthy_example_position,
    get_healthy_example_orientation,
)
from gaitmap.parameters.spatial_parameters import SpatialParameterCalculation

stride_list = get_healthy_example_stride_events()
positions = get_healthy_example_position()
orientations = get_healthy_example_orientation()

# %%
# Preparing the data
# ------------------
# Orientation and position are sampled at 100 Hz as they are derived from mocap data.
# The stride list is aligned with the IMU samples (204.8 Hz).
# Therefore, we need to convert the sampling rate for the stride list to be compatible with position and orientation.

stride_list["left_sensor"][["start", "end", "tc", "ic", "min_vel", "pre_ic"]] *= 100 / 204.8
stride_list["right_sensor"][["start", "end", "tc", "ic", "min_vel", "pre_ic"]] *= 100 / 204.8

# %%
# Creating SpatialParameterCalculation object
# -------------------------------------------
# We need this object for calculating the spatial parameters.

p = SpatialParameterCalculation()
p = p.calculate(stride_event_list=stride_list, positions=positions, orientations=orientations, sampling_rate_hz=100)

# %%
# Inspecting the results
# ----------------------
# The main output is the `parameters_`, which contains the spatial parameters for each stride in format of data frame
# in case of single sensor or dictionary of data frames for multiple sensors.
# As our passed stride_list here consists of two sensors, the output will be a dictionary.
p.parameters_["left_sensor"]

# %%
# `parameters_pretty_` is another version of `parameters_` but using human readable column names that indicate units.
p.parameters_pretty_["left_sensor"]
