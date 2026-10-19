# Accepted HMM upgrade and follow-up scope

This records the maintainer's decisions from the implementation conversation for
review of the three stacked PRs. These decisions explicitly authorize the stated
public changes under `.agents/refactor-policy.md`; they are not a general waiver.

## PR89: HMM replacement

Source: https://github.com/mad-lab-fau/gaitmap/pull/89

- Replace the used pomegranate subset with NumPy/Numba, rather than adopting a
  Torch dependency. Keep gaitmap's high-level workflows; small numerical training
  differences are allowed. Existing gaitmap JSON models must remain loadable.
- A native numeric interface may replace pomegranate's low-level interface. Put
  old-file conversion in the loader. Preserve pretrained inference regressions.
- Improve singular-covariance/tiny-probability handling rather than reproducing
  old numerical failures. Expose covariance regularization and training history.
- Optimize sparse transitions. Estimate composition connections from labeled
  instances only; do not add manual graph restrictions in the composition API.
- Keep fitted numeric HMMs and Gaussian mixtures as values. Use the existing tpcp
  wrappers for tuning; the maintainer explicitly chose this on 2026-10-19.
- The custom hierarchical example trains three multi-state submodels, using both
  feet in the foot body frame. Everything outside supplied strides is transition/
  null. Warn against stance/swing use without further validation. Prevent shared
  train/test observations and preprocessing leakage. Exclude missing annotations.
- Drop Python 3.9/3.10 and support 3.11 through 3.14. Test all four minors on Linux
  and Windows and lowest-direct dependency resolution on Python 3.11.
- Keep scientific/public-interface correctness and regression tests. Remove direct
  old-backend comparisons; retain compatibility tests for supported saved files.
- Keep usage guidance in the user guide/examples and API details in docstrings.
  The changelog explains the LLM-derived implementation and revalidation needs.

## PR91: Python/pandas modernization and sample coordinates

Source: https://github.com/mad-lab-fau/gaitmap/pull/91

- Modernize typing, iteration, pandas and resource APIs without new public APIs.
- Accept tpcp's NaN-aware default scoring; require tpcp >=3.0.0. Fix score input
  mutation and zero-division behavior. This dependency change belongs in PR91.
- Issue #35: require actual integer dtypes for sample/event positions and trajectory
  sample indices, including empty schemas. Integral floats are intentionally
  rejected. Missing events use nullable integers. Signed/unsigned widths work.
- Identifiers, arbitrary metadata, physical quantities and raw time indices in
  seconds remain unrestricted by sample-position dtype validation. No silent
  rounding. Update producers, consumers, migration guidance and scientific tests.

## PR94: algorithm audit corrections

Source: https://github.com/mad-lab-fau/gaitmap/pull/94

- Fix physical-axis selection in ZUPT, gradient index preservation, stale event
  detector results, ROI boundary/identifier handling and empty region trajectories.
- Correct SHOE's gravity direction to the per-window mean in its original
  equations; document the measured effect and need to revalidate thresholds.
- Remove the all-zero magnetometer special case, as requested in PR94 review.
- Constrained DTW issue #93 needs further consideration. Finite composition #80,
  heading smoothing and changes of turning-angle definitions remain out of scope.

## Recorded verification and limits

Before the mainline rebase, the combined stack passed 2,701 tests with 41 skips
on each of minimum-dependency Python 3.11 and locked Python 3.14. Sphinx executed
the custom example and built with three existing warnings. The new mainline
changes are review setup only; the rebase preserves runtime source exactly.
Reviewers inspect evidence and do not run tests. Regression agreement on these
recordings does not establish scientific validity for other populations or tasks.
