Running evaluation challenges
=============================

The runnable challenges from ``gaitmap-challenges`` are now included in gaitmap.
They prescribe datasets, cross-validation splits, and scoring functions for comparing algorithms.
The dataset files are downloaded separately and remain subject to their access conditions.
Global challenge configuration, saved-run management, and Bokeh visualization are not included.

Install the dataset loaders with::

    pip install "gaitmap[challenges]"

The current ``gaitmap-datasets`` release supports Python 3.11 through 3.13 for use with gaitmap.
On Python 3.14 and newer, the extra does not install the loaders.
Gaitmap algorithms and evaluation helpers can be used without these loaders.

Challenge locations
-------------------

* ``gaitmap.stride_segmentation.challenges`` provides stride segmentation challenges.
* ``gaitmap.parameters.challenges`` provides spatial parameter challenges.
* ``gaitmap.pipeline.challenges`` provides full-pipeline challenges.

Each namespace exports descriptively named challenge classes. Dataset-specific modules also expose
``Challenge``, ``ChallengeDataset``, and ``final_scorer``, as in the original package.
For example::

    from tpcp import Pipeline
    from tpcp.optimize import DummyOptimize
    from gaitmap.stride_segmentation.challenges import EgaitSegmentationValidation2014Challenge
    from gaitmap.stride_segmentation.challenges.egait_segmentation_validation_2014 import ChallengeDataset

    dataset = ChallengeDataset(data_folder="/path/to/dataset")
    challenge = EgaitSegmentationValidation2014Challenge(dataset=dataset)
    pipeline = ...  # A tpcp Pipeline that writes stride_list_ when run on one recording
    challenge.run(DummyOptimize(pipeline))
    results = challenge.cv_results_

Pass a trainable optimizer instead of ``DummyOptimize`` to train within each fold.
``cv_iterator`` controls the split iterator, and ``cv_params`` supplies additional ``tpcp.cross_validate`` options.
The predefined grouped and stratified challenges use ``tpcp.validate.DatasetSplitter`` to keep participant groups
and stratification labels attached to the correct dataset subset during nested optimization.

Input and output contracts
--------------------------

A datapoint is a single-recording subset of a ``tpcp.Dataset``.
Challenge helpers such as ``get_imu_data`` select the expected sensors from that recording.
The scorer supplies the datapoint to the pipeline's ``run`` method and compares its result with reference data.
Segmentation challenges expect ``stride_list_``. Spatial parameter challenges expect ``parameters_`` containing
per-sensor, per-stride gait parameters. Full-pipeline challenges expect ``aggregated_gait_parameters_``, a Series
of mean gait parameters for the recording.

The algorithm-specific ``evaluation`` modules also provide the matching, error calculation, and pooled score
aggregators for use in custom evaluations.
The integrated Kluge2017 challenge joins temporal and spatial reference columns for each stride before averaging them.
Its ``run`` method and the sensor-position full-pipeline challenge return the challenge instance.
These correct inconsistencies in the original challenge implementation.
Sensor-position parameter and full-pipeline challenges calculate temporal and spatial references over the same
IC-to-IC intervals, slicing the continuous marker data at those borders and including the endpoint sample.
This corrects a reference-interval mismatch in the original package and can change benchmark scores.
The affected sensor-position parameter and full-pipeline challenges therefore use ``VERSION = "2.0.0"``.
Re-run those benchmarks before comparing results across versions.

Migration from gaitmap-challenges
---------------------------------

Replace ``gaitmap_challenges.stride_segmentation`` with ``gaitmap.stride_segmentation.challenges``,
``gaitmap_challenges.spatial_parameters`` with ``gaitmap.parameters.challenges``, and
``gaitmap_challenges.full_pipeline`` with ``gaitmap.pipeline.challenges``.
Use ``cv_results_`` directly instead of the old result-storage helpers.
Existing ``gaitmap.evaluation_utils`` imports remain supported; new code should import the algorithm-specific
``evaluation`` module.
