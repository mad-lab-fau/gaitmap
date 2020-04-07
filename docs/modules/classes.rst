.. _api_ref:

=============
API Reference
=============

This is the API Reference for gaitmap sorted by algorithm types.

:mod:`gaitmap.base`: Base classes and utility functions
=======================================================

.. automodule:: gaitmap.base
    :no-members:
    :no-inherited-members:

Base classes
------------
.. currentmodule:: gaitmap.base

.. autosummary::
   :toctree: generated/
   :template: class_with_private.rst

    BaseAlgorithm
    BaseEventDetection
    BaseOrientationEstimation


:mod:`gaitmap.event_detection`: Algorithms to find temporal gait events
========================================================================

.. automodule:: gaitmap.event_detection
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: gaitmap.event_detection

.. autosummary::
   :toctree: generated/
   :template: class.rst

    RamppEventDetection


:mod:`gaitmap.trajectory_reconstruction.orientation_estimation`: Algorithms to estimate IMU sensor orientation given the sensor data
====================================================================================================================================
.. automodule:: gaitmap.trajectory_reconstruction.orientation_estimation
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: gaitmap.trajectory_reconstruction.orientation_estimation
.. autosummary::
   :toctree: generated/
   :template: class.rst

   GyroIntegration


:mod:`gaitmap.temporal_parameters`: Algorithm to calculate temporal parameters
==============================================================================

.. automodule:: gaitmap.temporal_parameters
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: gaitmap.temporal_parameters

.. autosummary::
   :toctree: generated/
   :template: class.rst

    TemporalParameterCalculation

Functions
---------
.. currentmodule:: gaitmap.temporal_parameters

.. autosummary::
   :toctree: generated/
   :template: function.rst

    TemporalParameterCalculation.calculate


:mod:`gaitmap.utils.rotations`: Helper to handle rotations
==========================================================

.. automodule:: gaitmap.utils.rotations
    :no-members:
    :no-inherited-members:

Functions
---------
.. currentmodule:: gaitmap.utils.rotations

.. autosummary::
   :toctree: generated/
   :template: function.rst

    rotation_from_angle
    rotate_dataset
    find_shortest_rotation
    get_gravity_rotation

:mod:`gaitmap.utils.vector_math`: Helper functions for finding shortest rotation
================================================================================

.. automodule:: gaitmap.utils.vector_math
    :no-members:
    :no-inherited-members:

Functions
---------
.. currentmodule:: gaitmap.utils.vector_math

.. autosummary::
   :toctree: generated/
   :template: function.rst

    is_almost_parallel_or_antiprallel
    normalize
    find_random_orthogonal
    find_orthogonal
    find_unsigned_3d_angle

:mod:`gaitmap.utils.array_handling`: Helper to perform various array modifications
==================================================================================

.. automodule:: gaitmap.utils.array_handling
    :no-members:
    :no-inherited-members:

Functions
---------
.. currentmodule:: gaitmap.utils.array_handling

.. autosummary::
   :toctree: generated/
   :template: function.rst

    sliding_window_view
    bool_array_to_start_end_array

:mod:`gaitmap.utils.dataset_helper`: Helper to perform validation of the default datatypes
==========================================================================================

.. automodule:: gaitmap.utils.dataset_helper
    :no-members:
    :no-inherited-members:

Functions
---------
.. currentmodule:: gaitmap.utils.dataset_helper

.. autosummary::
   :toctree: generated/
   :template: function.rst

    is_single_sensor_dataset
    is_multi_sensor_dataset
    get_multi_sensor_dataset_names
    is_single_sensor_stride_list
    is_multi_sensor_stride_list

:mod:`gaitmap.utils.coordinate_conversion`: Convert axes from sensor frame to body frame
========================================================================================

.. automodule:: gaitmap.utils.coordinate_conversion
    :no-members:
    :no-inherited-members:

Functions
---------
.. currentmodule:: gaitmap.utils.coordinate_conversion

.. autosummary::
   :toctree: generated/
   :template: function.rst

    convert_left_foot_to_fbf
    convert_right_foot_to_fbf
    convert_to_fbf

:mod:`gaitmap.utils.consts`: Global constants
=============================================

.. automodule:: gaitmap.utils.consts
    :member-order: bysource

:mod:`gaitmap.utils.static_moment_detection`: Helper to find static moments in sensor signals
=============================================================================================

.. automodule:: gaitmap.utils.static_moment_detection
    :no-members:
    :no-inherited-members:

Functions
---------
.. currentmodule:: gaitmap

.. autosummary::
   :toctree: generated/
   :template: function.rst

    utils.static_moment_detection.find_static_samples
    utils.static_moment_detection.find_static_sequences
