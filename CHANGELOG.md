# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) (+ the Migration Guide section),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# [1.2.0] - 

### Added

- Implemented local warping constraints for DTW. This should help in cases were only parts of a sequence are matched
  instead of the entire sequence.
  These constraints are available for `BaseDTW` and `BarthDTW`, but are **off** by default.
  A new class `ConstrainedBarthDTW` was added, that has the constraints **on** by default.
- Added support for `MultiSensorStrideLists` for all evaluation functions in `evaluation_utils.stride_segmentation`.
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/91)
- Global validation errors for region of interest list
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/88)
- Global validation errors for orientation/position/velocity lists
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/88)

### Changed

### Deprecated

### Removed

### Fixed

### Migration Guide


## [1.1.1] - 

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Migration Guide



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
 window size
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
