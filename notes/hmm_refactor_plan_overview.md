# HMM Refactor Plan Overview

This note replaces the earlier single investigation note with a staged plan.

The refactor should happen in three steps:

1. Move the current HMM system to a unified input config while still storing and operating on `pomegranate` models.
2. Introduce a generic serializable `HMMState` and change `RothSegmentationHmm.model` to use that state.
3. Add a dedicated `pomegranate 0.14` backend abstraction that converts between backend-native models and `HMMState`.

Why this order:

- Step 1 changes the public construction/configuration surface without changing the trained-model representation.
- Step 2 changes the trained-model representation while still keeping the current implementation behavior.
- Step 3 isolates the legacy backend after the public config and model-state surfaces are already stable.

This order reduces risk and keeps regressions easier to localize.
