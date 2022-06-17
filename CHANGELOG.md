# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) (+ the Migration Guide and 
Scientific Changes section), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For more information see the 
[Gitlab Releases Page](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/releases) of this 
project.

## [Unreleased]

### Added

- Data Transforms are added as a new algorithm class.
  These are simple algorithms that can take a single sensor data as input and provide a `transformed_data_` output.
  Transforms can be used for normalisations or feature transforms.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/148)
- A new helper function `gaitmap.utils.array_handling.iterate_region_data` is added.
  It allows to iterate over the data of multiple strides or ROIs cut from a dataset.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/149)
- A new helper (`gaitmap.utils.datatype_helper.to_dict_multi_sensor_data`) to convert DataFrame based MultiSensor data 
  to Dict-based MultiSensor data is added.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/149)
- A new example (`stride_segmentation/barth_dtw_custom_template.py`) is added, which explains how to create custom 
  dtw-templates from data.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/149)

### Changed

- All Algorithm classes now build on `tpcp.Algorithm`.
  This is a major change and might result in some unexpected incompatibilities with older code.
  The core functionality of tpcp should still work as expected.
  If you were using any of the base classes and algorithm helpers, check if they are still available in gaitmap.
  If not, there is likely a replacement in tpcp.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/173)
- The way DTW - templates are generated has been changed completely.
  Generation is now handled via BaseClasses instead of helper functions.
  These baseclasses follow the "optimizable" interface introduced in `tpcp`.
  Further, all templates now use the new data transforms instead of a fixed scaling factor.
  In combination, it should be much easier to create new templates and use more complex data normalisations.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/149)

### Removed

- We removed the entire `gaitmap.future` package.
  This was marked experimental anyway.
  All functionality has been moved into the `tpcp` package.
  For most methods and functions, it should be enough to just change the import statement.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/173)
- We removed `create_dtw_template` and `create_interpolated_dtw_template` as ways to create templates.
  Instead, you should now use `DtwTemplate` and `InterpolatedDtwTemplate` classes directly.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/149)

### Migration Guide

- If you were creating custom templates, you should now use the `DtwTemplate` and `InterpolatedDtwTemplate` classes.
  Check the new example (`stride_segmentation/barth_dtw_custom_template`) for more information, on how to use these new classes.
- If you were using scaling factors for DtwTemplates you should now use the new transformers in `tpcp.data_transforms`
  instead.
  To replace a fixed scaling factor, use `FixedScaler(scale=500)`.
  However, in many cases you might be better off using one of the Trainable scalars to automatically learn the scaling 
  factors from the template.
- If you worked with the raw data of the `BarthDtwTemplate`, be aware that the stored data is now **unscaled**
  (i.e. multiplied by 500 compared to the old version).


### Scientific Changes


## [1.6.0] - 2022-05-31

### Scientific Changes

- We fixed a bug with the sign of the TC and IC angle.
  They were flipped for real sensor data.
  Now they correctly follow the conventions [Kanzler2015](https://pubmed.ncbi.nlm.nih.gov/26737518/)
  This is a breaking change and your angle results will be different!
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/180)

## [1.5.0] - 2022-05-31

### Added

- A new submodule for ZUPT-Detection Algorithms and a first implementation in form of the `NormZuptDetector`.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/167)
- New sub module in preprocessing for sensor alignment functions
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/161)
- New algorithm `PcaAlignment` for sensor heading alignment using 2D PCA to align sensor frame to medio-lateral plane
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/161)
- New position method `PieceWiseLinearDedriftedIntegration` using piece wise linear dedrifted integration utilizing zupt
  updates.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/166)
- A new algorithm (`ForwardDirectionSignAlignment`) to detect the walking direction to automatically align the sensor 
  attachment orientation.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/171)
- An example that explains how to automatically align a sensor to the gaitmap coordinate system, when it was attached in 
  an arbitrary orientation.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/171)
- The `HerzerEventDetection` - Algorithm, specifically designed to work with stair gait.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/158)

### Changed

- Cloning is now more robust and falls back to deep cloning objects that are not algorithms.
  This prevents issues when mutable objects (other than nested algorithms) are used in parameters
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/170)
- We now handle mutable defaults to prevent strange edge cases! Read more about it in the `project_structure` guide
  under the development-guide tab in the Documentation.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/168)
- The constrained DTW only tracks a single counter now.
  By using the positive and the negative value range, we can save 1/3 of the overall RAM.
  This should improve performance.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/155)
- `RtsKalman.find_zupts` now gets passed the entire data **before** any unit conversions.
  Before it only got the gyro data already converted in rad/s.
  This change should make it easier to overwrite the ZUPT detection with custom methods.
  This is a breaking change! See the migration guide for more details.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/159)
- The Zupt-Detection of `RtsKalman` is now handled via dependency injection.
  Instead of specifying a list of parameters for the Zupt detection you can simply pass a instance of a Zupt detector.
  (see `gaitmap.zupt_detection`).
  The old parameters a deprecated.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/169)
- The `memory` option for `RtsKalman` was removed.
  This is a breaking change!
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/159)

### Deprecated

- The parameters `"zupt_threshold_dps", "zupt_window_length_s", "zupt_window_overlap_s"` for `RtsKalman` are now 
  deprecated.
  Instead, a Zupt-Detector instance should be used.
  When converting from the current parameters to the new `NormZuptDetector` class, note that the overlap is now 
  specified as a fraction of the window length and not in seconds!
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/159)


### Removed

- The `memory` option for `RtsKalman` was removed.
  This is a breaking change!
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/159)

### Fixed

- Bugfix for `gaitmap.gait_detection.ullrich_gait_sequence_detection`: Now proper handling of cases where margin
  should be added but no gait sequences were detected 
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/157) 

### Migration Guide

- We removed the orientation update feature of the `RtsKalman` class.
  This feature was off by default.
  Unless you explicitly enabled it, you should be fine.
  In case you used the feature, remove all references to the respective keywords.
- The call signature and the units of the gyro data passed `RtsKalman.find_zupts` has changed.
  In case you where using custom subclasses of `RtsKalman` to implement custom zupt methods, you need to update your 
  zupt methods for the new input structure.
  However, due to the new changes to ZuptDetectors, it is likely that you do not need a custom subclass anymore, but can
  simply pass a instance of a ZuptDetector to the Kalman class
- The `memory` option for `RtsKalman` was removed, as it was not particularly useful. In case you need caching we 
  recommend implementing it externally.

### Scientific Changes

- The "ori_update" feature of the Kalman filter was removed again. 
  The feature was experimental anyway, but in turns out the implementation was actually wrong.
  The RTSKalman filter tracks the change in error state open-loop.
  This means errors are not applied during ZUPT-measurements, but simply added up and corrected at the very end.
  This works well for position and velocity, but the equations for orientation assume, that the orientation error 
  remains small enough so that a linearization and separation of the angle error on the different axis is possible.
  When the filter is run open loop, it is very likely that the orientation error will quickly reach ranges, where this 
  doesn't hold anymore.
  We concluded that the used method to track the orientation error (and hence, correct it directly) is not suitable for 
  open-loop Kalmanfilter designs.
  It is unlikely that the feature will be added again in the future.
  However, prior versions can be used as reference to implement a closed loop Kalmanfilter with direct observations of 
  the orientation error.
  This is a breaking change!
  See the Migration section for more details, if you used this feature before.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/159)


## [1.4.0] - 2021-06-25

With 1.4. we finally realize the longterm goal to make the comparison of algorithms easy and unify code across dataset 
and algorithms.
The highlights of this release are the new experimental `Dataset` and `Pipeline` classes and first-class support for
trainable algorithms.

### Added

- New module for experimental features. Things implemented there might be changed or deleted without deprecation
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/122)
- New dataset class that will serve as the base class for all future datasets in `gaitmap.future`. 
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/122,
  https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/132)
- New example on how to create custom datasets with the new dataset-class
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/132)
- The new gaitmap logo is now shown everywhere :)
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/129)
- New parameter `zero_division` added to all methods in `gaitmap.evaluation_utils.scores` allowing for setting the
  return value in case of a zero_division happening. 
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/134)
- A new interpolation method `gaitmap.utils.array_handling.multi_array_interpolation` to interpolate multiple 2D arrays 
  at once to the same length.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/138)
- Baseclasses to build custom pipelines are added
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/128)
- A GridSearch optimizer (similar to GridSearchCV in sklearn, but without the CV part) with a respective example is
  added. (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/128)
- A generic wrapper for algorithms and pipelines that can be optimized/trained with respective example.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/128)
- Added a warning about providing multiple methods when using any trajectory wrapper class to prevent user error.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/142)
- Added a new `safe_run` method to pipelines that should be used over the normal `run` method.
  `safe_run` performance multiple checks to catch potential implementation errors in user created pipelines.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/144)
- New parameter `enforce_consistency` added to `gaitmap.event_detection.RamppEventDetection` to make event detection 
  postprocessing optional.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/136)
- A GridSearchCV optimizer (similar to GridSearchCV in sklearn) with respective example is added.
  This version is specifically implemented for gaitmap and has some fancy ways of performance optimization.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/139)
- A cross-validation function is added to evaluate optimizable pipelines of any kind.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/139)
- `gaitmap.gait_detection.ullrich_gait_sequence_detection.UllrichGaitSequenceDetection` now has the option to add a
 margin that will be symmetrically added around the detected gait sequences. This should help to included intervals
  before and after steady state gait, that can be missed by the algorithm.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/152)

### Changed

- Changed the output format of `gaitmap.evaluation_utils.scores.precision_recall_f1_score` so it will
  be easier to use with the pipelines and optimizer.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/133)
- Made parameters for `gaitmap.evaluation_utils` keyword only to avoid input confusion.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/135)
- Improved the performance of dtw template interpolation by a factor of 15-30 (depending on the interpolation type).
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/138)
- File based dtw templates do not implicitly cache the template data anymore.
  This means calling `get_data` on the template does not change the object anymore, which is desirable in the context of
  caching.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/143)

### Deprecated

### Removed

- `gaitmap.utils.array_handling.interpolate1d` was removed as it is not needed anymore internally.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/138)

### Fixed
- After internal updates of scipy>=1.6 we now manually handle empty orientation inputs in 
`gaitmap.parameters.spatial_parameters._calc_turning_angle` and 
`gaitmap.parameters.spatial_parameters._compute_sole_angle_course`.
(https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/153, 
https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/154)

### Migration Guide

- If you used `gaitmap.evaluation_utils.scores.precision_recall_f1_score`, the output is now a dictionary instead of a
  tuple. Migration should be straight forward.
- Some parameters are now keyword only. In case you get an unexpected error, telling you that "an algorithm only 
  receives n positional arguments, but you provided n+1", double-check the call signature.

### Scientific Changes



## [1.3.0] - 2021-01-25

### Added

- New evaluation function to compare the output of temporal or spatial parameter calculation with ground truth.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/105)
- New function for merging intervals that are overlapping and that are within a specific distance from each other in
  `utils.array_handling`. (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/120)
- Some algorithms now support caching intermediate results with `joblib`.
  There is also a new example explaining caching in more detail.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/126 (mostly reverted),
  https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/127)
- Experimental support for ZUPT based orientation updates in the `RtsKalman` filter.
  This approach is not fully validated and should be used with care.
  If the feature is turned on, a runtime warning will indicate that as well.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/113)
- `RtsKalman` has a new results `zupts_` that can be used to check which zero-velocity region were used by the filer.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/113)
- A new guide on how to evaluate (gait)parameters using parameter optimization and cross validation.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/124)

### Changed

- `evaluation_utils.stride_segmentation._match_label_lists` now works for n-D and should be faster for large stride 
  length. (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/110)
- Internal refactoring of how the algorithms combine the results of the single sensors to the final output.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/116)
- Starting from this release the name "sensor data" instead of "dataset" is used to refer to data from multiple sensors
  within one recording.
  The term "dataset" will be used in the future to describe entire sets of multiple recordings potentially including
  multiple participants and further information besides raw IMU data.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/117)
- Internal refactoring that should improve typing. `mypy` now only returns a couple of issues that are not easily
  fixable or actual `mypy` bugs.
  They will be resolved step by step in the future.
  mypy checking can be done using the new command `dodo type_check` and developer should try to not introduce further
  typing issues.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/118)
  The refactoring also required some small user facing changes:
  - The `regions_of_interest` parameter of `RoiStrideSegmentation.segment` is now keyword only to be compatible with the
    `StrideSegmentation` base class.
- The `UllrichGaitSequenceDetection` now returns a dict also in the case of merged gait sequences. 
  The gait sequences stored for all sensor are set equal. 
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/121)
- `gait_detection.ullrich_gait_sequence_detection` now uses `utils.array_handling.merge_intervals` and should therefore
  be able to merge sequences faster. 
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/120)
- The `regions_of_interest` parameter of `RoiStrideSegmentation.segment` is now keyword only to be compatible with the
  `StrideSegmentation` base class.
- `RtsKalman.covariance` now has the shape `(len(data), 9 * 9)` instead of `(len(data) * 9, 9)` which should make it 
  easier to plot.
  

### Deprecated

### Removed
    
- `gait_detection.ullrich_gait_sequence_detection._gait_sequences_to_boolean_single` has been made redundant by 
  `utils.array_handling.merge_intervals` and was therefore removed. 
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/120)
- Removed old serialization format for DTW templates that was deprecated a couple of releases ago.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/issues/127)  

### Fixed

- In `stride_segmentation.roi_stride_segmentation` the assignment of the offset corrected strides to the
 `combined_stride_list` was fixed.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/114)
- Added a note to BarthDTW to clarify the calculation of the distance matrix.
- Changed the Chebyshev distance function in `evaluation_utils.stride_segmentation._match_label_lists` to a general one 
  that can use n-D input. (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/115)
- Fixed bug in `utils.static_moment_detection.find_static_samples` where extremely short sequences, which would only fit
  a single window, would fail.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/109)

### Migration Guide

- Due to the renaming of dataset to sensordata, a couple of names and import path are changed:
  - All imports from `gaitmap.utils.dataset_helper` should be changed to `gaitmap.utils.datatype_helper`
  - `SingleSensorDataset`, `MultiSensorDataset` and `Dataset` are now called, `SingleSensorData`, `MultiSensorData` and
    `SensorData`, respectively.
  - The functions `is_single_sensor_dataset`, `is_multi_sensor_dataset`, `is_dataset`, and 
    `get_multi_sensor_dataset_names` are renamed to `is_single_sensor_data`, `is_multi_sensor_data`, `is_sensor_data`, 
    and `get_multi_sensor_names`.

### Scientific Changes


## [1.2.0] - 2020-11-11

### Added

- An Error-State-Kalman-Filter with rts smoothing (`RtsKalman`) was added that can be used to estimate orientation and
  position over longer regions by detecting its own ZUPTs.
  This can (should) be used in combination with `RegionLevelTrajectory`.
- A `RegionLevelTrajectory` class that calculates trajectories for entire ROIs/gait sequences.
  Otherwise the interface is very similar to `StrideLevelTrajectory`.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/87,
  https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/98,
  https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/85)
- Added a new example for `RegionLevelTrajectory`.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/87) 
- Support for trajectory methods that can calculate orientation and position in one go is added for 
  `StrideLevelTrajectory` (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/87)
- Implemented local warping constraints for DTW. This should help in cases were only parts of a sequence are matched
  instead of the entire sequence.
  These constraints are available for `BaseDTW` and `BarthDTW`, but are **off** by default.
  A new class `ConstrainedBarthDTW` was added, that has the constraints **on** by default.
- Added support for `MultiSensorStrideLists` for all evaluation functions in `evaluation_utils.stride_segmentation`.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/91)
- Global validation errors for region of interest list
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/88)
- Global validation errors for orientation/position/velocity lists
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/88, 
  https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/95)
- New evaluation functions to compare the results of event detection methods with ground truth
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/92)
- Added the functionality to merge gait sequences from individual sensors in `UllrichGaitSequenceDetection` 
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/93)

### Changed

- The initial orientation for the Stride and Region Level trajectory now uses the correct number of samples as input.
  This might change the output of the integration method slightly!
  However, it also allows to pass 0 for `align_window_length` and hence, just use the first sample of the integration
  region to estimate the initial orientation.
  (mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/98)
- It is now ensured that in **all** in gaitmap where a start and an end sample is provided, the end sample is
  **exclusive**.
  Meaning if the respective region should be extracted from a dataarray, you can simply do `data[start:end]`.
  All functions that used different assumptions are changed.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/97~,
  https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/103)
- The default check for orientation/position/velocity lists now only expects a "sample" column and not a "sample" and a
  "s_id" column by default.
  The `s_id` parameter was removed and replaced with a more generic `{position, velocity, orientation}_list_type`
  parameter than can be used to check for either stride or region based lists.
  See the migration guide for more information.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/88)
- Internal restructuring of the evaluation utils module.
  In case you used direct imports of the functions from the submodule, you need to update these.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/92)

### Deprecated

### Removed

### Fixed

- Fixed possible `ZeroDivisionError` when using `evaluation_utils.scores`.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/104)

### Migration Guide

- In case your algorithm uses `is_{single,multi}_sensor_{position, velocity, orientation}_list` all usages must be 
  updated to include `{position, velocity, orientation}_list_type="stride"` to restore the default behaviour before the
  update.
  If you were using the function with `s_id=False`, you can update to `{position, velocity, orientation}_list_type=None`
  to get the same behaviour.

### Scientific Changes

- All Dtw based methods now produce outputs that follow our index guidelines.
  This means that they correctly produce matches, which end index is **exclusive**.
  In some cases this might change existing results by adding 1 to the end index.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/103)
- The initial orientation in the StrideLevelTrajectory now has a window with exactly the number of samples specified
  (instead of one less due to Python indexing).
  This will lead to very small changes in the calculated trajectory.
  (mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/98)


## [1.1.0] - 2020-10-04

### Added

- Proper support for gait sequences and regions of interest in the stride segmentation
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/64, 
  https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/71,
  https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/75).
  Note that support was first implemented in the segmentation algorithms directly and then reworked and moved into the
  `RoiStrideSegmentation` wrapper class.
- "Max. lateral excursion" and "max. sensor lift" are now implemented as two new spatial parameters.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/79)
- The `align_heading_of_sensors` method has now an option to run an additional smoothing filter that can avoid
  misalignments in certain cases.
- `VelocityList` is now a separate dtype representing integrated velocity values
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/77)
- `UllrichGaitSequenceDetection` now has its own example in the Sphinx Gallery 
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/83)
- Added a new module `gaitmap.utils.signal_processing` for general filtering and processing purposes (mainly for
  moving some functions from `UllrichGaitSequenceDetection` 
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/83)


#### The new `evaluation_utils` module https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/issues/117

A new module was started that will contain helper functions to evaluate the results of gait analysis pipelines.
This is the first version and we already added:

- A set of function to compare stride lists based on start and end values 
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/66)
- A set of common metrics (recall, precision, f1) that can be calculated for segmentation results
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/66)
  
#### Global Validation Error Handling

https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/issues/72,
https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/issues/16

To make developing new algorithms easier and error messages easier to understand for users,
we modified the data-object validation helper to raise proper descriptive error messages.
The data-object helpers now have an optional argument (`raise_exception=True`) to trigger the new error-messages.

- Proper validation for the dataset objects.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/80)
- A new `is_dataset` method to validate either single or multi-sensor datasets.
  This should be used in functions that should work with both inputs.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/80)
- Proper validation for the stride list objects.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/86)
- A new `is_stride_list` method to validate either single or multi-sensor stride lists.
  This should be used in functions that should work with both inputs.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/86)
- A custom error class `ValidationError` used for all validation related errors.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/80)

All existing algorithms are updated to use these new validation methods.
(https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/80,
https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/84,
https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/86)

The remaining datatypes will get updated validation functions in a future release.

### Changed

- To get or load the data of a DtwTemplate you now need to call `get_data` on the method instead of just using the
  `data` property (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/73)
  This is done to conform with the basic class structure needed for proper serialization.
- Changed function name `_butter_lowpass_filter` in `gaitmap.gait_detection.UllrichGaitSequenceDetection` to 
`butter_lowpass_filter_1d` (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/83)
- Moved `butter_lowpass_filter_1d` and `row_wise_autocorrelation` from `gaitmap.gait_detection
.UllrichGaitSequenceDetection` to the new module `gaitmap.utils.signal_processing`
(https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/83)

### Deprecated

- The format in which DataFrame Attributes are stored in json has been changed.
  The old format can still be loaded, but this will be removed in future versions.
  Related to https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/72.
  See the migration guide for actions to take. 

### Removed

### Fixed

- Fixed a bug in the madgwick algorithms that might have caused some incorrect results in earlier version
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/70)
- Fixed issue that templates that were stored in json do not preserve order when loaded again (see more info in migration guide)
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/72)
- Fixed an issue that `rotate_dataset_series` performed an unexpected inplace modification of the data.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/78)
- Fixed a bug that would break the `UllrichGaitSequenceDetection` in case of no active signal windows
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/commit/95736e8d2676f98d4c43ea0bfa3dbf3566542f71)
- Fixed a bug that would break the `UllrichGaitSequenceDetection` in case the input signal was just as long as the
  window size.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/commit/d516a40520f86bfd39ddcdd813b5c18312785085)
- Adapted scaling factors for the usage of accelerometer data in the `UllrichGaitSequenceDetection` to work with
  values given in m/s^2 (in contrast to g as done in the publication / the usage with mGL-algorithms-py)
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/83)
- Fixed install link in project README

### Migration Guide

- The format in which DataFrame Attributes are stored in json has been changed and the old format will be fully removed
  in the future.
  If you have DtwTemplates stored as json, load them (you will get a warning) once. Double check that the ordering of 
  the template data is correct. If it is not, sort it correctly and then save the object again.
  This should fix all issues going forward.
- To get or load the data of a DtwTemplate, you now need to use `template.get_data()` instead of `template.data`.
  Simply replacing the old with the new option should fix all issues.
- For custom algorithms using the gaitmap format, you should migrate to the new datatype validation functions!
  See "Added" section for more details.

## [1.0.1] - 2020-07-15

A small bug fix release that fixes some issues with the new template creation function.

### Fixed

- The stride interpolation template creation function now works correctly if the index of the individual strides is 
  different (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/68).
- Templates that use datafraes to store the template data can now be serialized
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/69)

## [1.0.0] - 2020-06-26

This is the first official release of gaitmap (Wuhu!).
Therefore, we do not have a proper changelog for this release.

### Added
- All the things :)

### Changed

### Deprecated

### Removed

### Fixed

### Migration Guide
