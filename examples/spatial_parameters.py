r"""
Spatial parameters calculation
===============================

This example illustrates illustrates how spatial parameters can be calculated for each stride  by
the :class:`~gaitmap.parameters.spatial_parameters.SpatialParameterCalculation`.
The used implementation is based on the work of Kanzler et al [1]_.

.. [1] Kanzler, C. M., Barth, J., Rampp, A., Schlarb, H., Rott, F., Klucken, J., Eskofier, B. M. (2015, August).
       Inertial sensor based and shoe size independent gait analysis including heel and toe clearance estimation.
       In 2015 37th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)
       (pp. 5424-5427). IEEE.
"""

# %%
# Getting input data
# --------------------------
#
# For this we need stride event list obtained from event detection method, position and orientation list
# obtained from trajectory reconstruction.
from gaitmap.example_data import (
    get_healthy_example_stride_events,
    get_healthy_example_position,
    get_healthy_example_orientation,
)
from gaitmap.parameters.spatial_parameters import SpatialParameterCalculation

# sphinx_gallery_thumbnail_path = '_static/gait-iStock-517423987-copy-550x717.jpg'
stride_list = get_healthy_example_stride_events()
positions = get_healthy_example_position()
orientations = get_healthy_example_orientation()

# %%
# Creating an object of SpatialParameterCalculation and using it for calculating the spatial parameters.

p = SpatialParameterCalculation()
p = p = p.calculate(
    stride_event_list=stride_list, positions=positions, orientations=orientations, sampling_rate_hz=204.8
)

# %%
# Inspecting the results
# ----------------------
# The main output is the `parameters_`, which contains the spatial parameters for each stride in format of data frame
# in case of single sensor or dictionary of data frames for multiple sensors.
# As our passed stride_list here consists of two sensors, the output will be a dictionary.
p.parameters_["left_sensor"]
