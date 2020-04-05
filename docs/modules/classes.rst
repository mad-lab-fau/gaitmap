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
.. currentmodule:: gaitmap

.. autosummary::
   :toctree: generated/
   :template: class_with_private.rst

    base.BaseAlgorithm


:mod:`gaitmap.utils.rotations`: Helper to handle rotations
==========================================================

.. automodule:: gaitmap.utils.rotations
    :no-members:
    :no-inherited-members:

Functions
---------
.. currentmodule:: gaitmap

.. autosummary::
   :toctree: generated/
   :template: function.rst

    utils.rotations.rotation_from_angle
    utils.rotations.rotate_dataset
    utils.rotations.find_shortest_rotation
    utils.rotations.get_gravity_rotation

:mod:`gaitmap.utils.vector_math`: Helper functions for finding shortest rotation
================================================================================

.. automodule:: gaitmap.utils.vector_math
    :no-members:
    :no-inherited-members:

Functions
---------
.. currentmodule:: gaitmap

.. autosummary::
   :toctree: generated/
   :template: function.rst

    utils.vector_math.is_almost_parallel_or_antiprallel
    utils.vector_math.normalize
    utils.vector_math.find_random_orthogonal
    utils.vector_math.find_orthogonal
    utils.vector_math.find_unsigned_3d_angle

:mod:`gaitmap.utils.array_handling`: Helper to perform various array modifications
==================================================================================

.. automodule:: gaitmap.utils.array_handling
    :no-members:
    :no-inherited-members:

Functions
---------
.. currentmodule:: gaitmap

.. autosummary::
   :toctree: generated/
   :template: function.rst

    utils.array_handling.sliding_window_view

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

:mod:`gaitmap.utils.consts`: Global constants
=============================================

.. automodule:: gaitmap.utils.consts
    :member-order: bysource
