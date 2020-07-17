# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) (+ the Migration Guide section),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] -

### Added

#### The new `evaluation_utils` module [#117](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/issues/117)

A new module was started that will contain helper functions to evaluate the results of gait analysis pipelines.
This is the first version and we already added:

- A set of function to compare stride lists based on start and end values [!66](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/66)
- A set of common metrics (recall, precision, f1) that can be calculated for segmentation results [!66](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/66)

### Changed

### Deprecated

### Removed

### Fixed

- Fixed a bug in the madgwick algorithms that might have caused some incorrect results in earlier version
  (https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/merge_requests/70)

### Migration Guide

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