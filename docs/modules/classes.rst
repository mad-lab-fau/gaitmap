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

:mod:`gaitmap.utils.consts`: Global constants
=============================================

.. automodule:: gaitmap.utils.consts
    :member-order: bysource
