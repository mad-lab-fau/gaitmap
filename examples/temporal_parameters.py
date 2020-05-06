r"""
Temporal parameters calculation
===============================

This example illustrates how temporal parameters can be calculated for each stride  by
the :class:`~gaitmap.parameters.temporal_parameters.TemporalParameterCalculation`.
"""

# %%
# Getting stride list
# --------------------------
#
# For this we need stride event list that can be obtained from event detection method.
from gaitmap.example_data import get_healthy_example_stride_events
from gaitmap.parameters.temporal_parameters import TemporalParameterCalculation

stride_list = get_healthy_example_stride_events()

# %%
# Creating TemporalParameterCalculation object
# -------------------------------------------
# We need this object for calculating the temporal parameters.
# Temporal parameters are calculated based on ic and tc events

p = TemporalParameterCalculation()
p = p.calculate(stride_event_list=stride_list, sampling_rate_hz=204.8)

# %%
# Inspecting the results
# ----------------------
# The main output is the `parameters_`, which contains the temporal parameters for each stride in format of data frame
# in case of single sensor or dictionary of data frames for multiple sensors.
# As our passed stride_list here consists of two sensors, the output will be a dictionary.
p.parameters_["left_sensor"]
