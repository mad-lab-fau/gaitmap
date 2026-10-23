gaitmap.pipeline: Reusable gait analysis pipelines
==================================================

.. automodule:: gaitmap.pipeline
    :no-members:
    :no-inherited-members:

Classes
-------

.. currentmodule:: gaitmap.pipeline

.. autosummary::
   :toctree: generated/pipeline
   :template: class.rst

    GenericGaitPipeline
    BaseGaitDataset

The ``BaseGaitDataset`` protocol requires only ``data`` and ``sampling_rate_hz``.
A recording may be a single-recording ``tpcp.Dataset`` subset or an ordinary Python object.
Dataset loaders satisfy this interface structurally; they do not need to inherit from gaitmap or depend on it.

``GenericGaitPipeline.run`` consumes one such recording with data already aligned to the gaitmap foot sensor frame.
Pass stride segmentation and event detection algorithms to the constructor. Trajectory reconstruction and temporal
and spatial parameter calculation have defaults that can be replaced independently.
Every stage runs on a clone, and nested algorithm parameters are accessible through ``set_params``.
For single-sensor input, set ``single_sensor_side``. For multiple sensors, configure ``left_like`` and ``right_like``
to match the foot sensor names.

See :doc:`/auto_examples/full_pipelines/generic_gait_pipeline` for a runnable example.

For a challenge that selects sensors or renames them, adapt the recording through the challenge's ``get_imu_data``
helper. For example::

    from types import SimpleNamespace
    from gaitmap.pipeline import GenericGaitPipeline
    from gaitmap.pipeline.challenges import SensorPositionComparisonInstepChallenge

    class InstepPipeline(GenericGaitPipeline):
        def run(self, datapoint):
            recording = SimpleNamespace(
                data=SensorPositionComparisonInstepChallenge.get_imu_data(datapoint),
                sampling_rate_hz=datapoint.sampling_rate_hz,
            )
            return super().run(recording)

The challenge still passes the original dataset datapoint to its scorer, so reference labels remain available.

Evaluation
----------

.. automodule:: gaitmap.pipeline.evaluation
    :members:

Challenges
----------

See :doc:`challenges` for installation and usage.

.. currentmodule:: gaitmap.pipeline.challenges

.. autosummary::
   :toctree: generated/pipeline/challenges
   :template: class.rst

    Kluge2017Challenge
    SensorPositionComparisonInstepChallenge
