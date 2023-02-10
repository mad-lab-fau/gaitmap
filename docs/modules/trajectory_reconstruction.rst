gaitmap.trajectory_reconstruction: Algorithms to estimate IMU sensor orientation and position
====================================================================================================
.. automodule:: gaitmap.trajectory_reconstruction
    :no-members:
    :no-inherited-members:

TrajectoryWrapperClasses
------------------------
.. currentmodule:: gaitmap.trajectory_reconstruction
.. autosummary::
   :toctree: generated/trajectory_reconstruction
   :template: class.rst

    StrideLevelTrajectory
    RegionLevelTrajectory

Trajectory Estimation Methods
-----------------------------
.. automodule:: gaitmap.trajectory_reconstruction.trajectory_methods
    :no-members:
    :no-inherited-members:

.. currentmodule:: gaitmap.trajectory_reconstruction
.. autosummary::
   :toctree: generated/trajectory_reconstruction
   :template: class.rst

    RtsKalman
    MadgwickRtsKalman


Orientation Estimation Methods
------------------------------
.. automodule:: gaitmap.trajectory_reconstruction.orientation_methods
    :no-members:
    :no-inherited-members:

.. currentmodule:: gaitmap.trajectory_reconstruction
.. autosummary::
   :toctree: generated/trajectory_reconstruction
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
   :toctree: generated/trajectory_reconstruction
   :template: class.rst

    ForwardBackwardIntegration
    PieceWiseLinearDedriftedIntegration