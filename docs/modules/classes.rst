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
    BaseStrideSegmentation
    BaseEventDetection
    BaseOrientationMethod
    BasePositionMethod
    BaseTrajectoryReconstructionWrapper
    BaseTemporalParameterCalculation
    BaseSpatialParameterCalculation


:mod:`gaitmap.preprocessing`: Helper to align and prepare datasets
==================================================================

.. automodule:: gaitmap.preprocessing
    :no-members:
    :no-inherited-members:

Functions
---------
.. currentmodule:: gaitmap.preprocessing

.. autosummary::
   :toctree: generated/
   :template: function.rst

   align_dataset_to_gravity


:mod:`gaitmap.stride_segmentation`: Algorithms to find stride candidates
========================================================================

.. automodule:: gaitmap.stride_segmentation
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: gaitmap.stride_segmentation

.. autosummary::
   :toctree: generated/
   :template: class.rst

    BaseDtw
    BarthDtw
    DtwTemplate
    BarthOriginalTemplate

Functions
---------
.. currentmodule:: gaitmap.stride_segmentation

.. autosummary::
   :toctree: generated/
   :template: function.rst

    create_dtw_template
    base_dtw.find_matches_find_peaks
    base_dtw.find_matches_min_under_threshold


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


:mod:`gaitmap.trajectory_reconstruction`: Algorithms to estimate IMU sensor orientation and position
====================================================================================================
.. automodule:: gaitmap.trajectory_reconstruction
    :no-members:
    :no-inherited-members:

TrajectoryWrapperClasses
------------------------
.. currentmodule:: gaitmap.trajectory_reconstruction
.. autosummary::
   :toctree: generated/
   :template: class.rst

    StrideLevelTrajectory

Orientation Estimation Methods
------------------------------
.. automodule:: gaitmap.trajectory_reconstruction.orientation_methods
    :no-members:
    :no-inherited-members:

.. currentmodule:: gaitmap.trajectory_reconstruction
.. autosummary::
   :toctree: generated/
   :template: class.rst

    SimpleGyroIntegration
    MadgwickAHRS

Position Estimation Methods
---------------------------
.. automodule:: gaitmap.trajectory_reconstruction.position_methods
    :no-members:
    :no-inherited-members:

.. currentmodule:: gaitmap.trajectory_reconstruction
.. autosummary::
   :toctree: generated/
   :template: class.rst

    ForwardBackwardIntegration


:mod:`gaitmap.parameters`: Algorithm to calculate biomechanical parameters
==========================================================================

.. automodule:: gaitmap.parameters
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: gaitmap.parameters

.. autosummary::
   :toctree: generated/
   :template: class.rst

    TemporalParameterCalculation
    SpatialParameterCalculation


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
    find_unsigned_3d_angle
    find_angle_between_orientations
    find_rotation_around_axis


:mod:`gaitmap.utils.vector_math`: Helper functions to deal with mathematical Vectors
====================================================================================

.. automodule:: gaitmap.utils.vector_math
    :no-members:
    :no-inherited-members:

Functions
---------
.. currentmodule:: gaitmap.utils.vector_math

.. autosummary::
   :toctree: generated/
   :template: function.rst

    is_almost_parallel_or_antiparallel
    normalize
    find_random_orthogonal
    find_orthogonal


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
    find_local_minima_below_threshold
    find_local_minima_with_distance
    find_extrema_in_radius
    split_array_at_nan


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
    is_single_sensor_orientation_list
    is_multi_sensor_orientation_list
    is_single_sensor_position_list
    is_multi_sensor_position_list
    set_correct_index


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


:mod:`gaitmap.utils.static_moment_detection`: Helper to find static moments in sensor signals
=============================================================================================

.. automodule:: gaitmap.utils.static_moment_detection
    :no-members:
    :no-inherited-members:

Functions
---------
.. currentmodule:: gaitmap.utils.static_moment_detection

.. autosummary::
   :toctree: generated/
   :template: function.rst

    find_static_samples
    find_static_sequences


:mod:`gaitmap.utils.consts`: Global constants
=============================================

.. automodule:: gaitmap.utils.consts
    :member-order: bysource


:mod:`gaitmap.example_data`: Helper to load some example data
==============================================================================

.. automodule:: gaitmap.example_data
    :no-members:
    :no-inherited-members:

Functions
---------
.. currentmodule:: gaitmap.example_data

.. autosummary::
   :toctree: generated/
   :template: function.rst

    get_healthy_example_imu_data
    get_healthy_example_stride_borders
    get_healthy_example_mocap_data
    get_healthy_example_stride_events
    get_healthy_example_orientation
    get_healthy_example_position
