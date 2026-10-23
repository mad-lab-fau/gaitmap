# Algorithm performance implementation

## Accepted scope

Implement the performance review findings in one PR based on
`10afaccc2d2e74dba239168c38e0be1a0877edc0`. Preserve numerical results, event
boundaries, tie handling, ordering, dtypes, public intermediate outputs and
persisted formats under `.agents/refactor-policy.md`.

The maintainer explicitly waived lateral-excursion warning equivalence. Other
warnings remain covered by the existing behavior. Private helper signatures may
change with all in-repository consumers updated. No documented public signature
changes are intended.

The pre-existing reciprocal-matching Chebyshev indexing defect is outside this
performance PR. The maintainer requested its correction in a separate PR based
on master. Do not fold that behavior change into this range.

## Implementation checklist

- DTW: linear retained-path selection and allocation-free traceback with the
  existing predecessor priority and NaN behavior.
- HMM: binary-search stride ends, reuse Viterbi score buffers, retain rolling
  forward rows when likelihood-only consumers discard history. Preserve full
  history for posterior and Baum-Welch statistics.
- Trajectories: reuse nonempty-region lookups, remove only redundant AHRS
  quaternion copies, reuse sigmoid weights, hoist the Kalman identity matrix,
  reuse forward-alignment rotations and piecewise sample coordinates.
- Detection: derive static masks from integer interval coverage and compute only
  Ullrich's retained autocorrelation lags using the existing arithmetic.
- Spatial parameters: reuse grouped endpoints and stride length, vectorize
  float64 lateral excursion, retain original arithmetic for other dtypes.
- Evaluation: reuse parameter-error arithmetic, keep small nearest-neighbor
  queries on one worker and allocate interval masks directly as boolean arrays.

No changes to constrained-DTW state semantics, Kalman transition-history
contracts, motion-update hooks, precision, numerical solvers, FFT methods,
floating-point reduction order or fastmath are included.

## Verification evidence

The implementation uses Python 3.11.13, NumPy 2.4.3, pandas 3.0.0, SciPy 1.17.1
and Numba 0.64.0 from an existing environment. `PYTHONPATH` points at this
checkout and `packages/gaitmap_mad/src`, so its editable installation does not
select another worktree.

- Evaluation and array helpers: 190 tests passed before and after editing.
  Exact baseline comparisons passed for 168 matching cases across both worker
  paths, seven numeric/nullable parameter dtypes and 320 interval-mask cases.
- Segmentation: baseline 327 passed, 7 skipped; final focused suite 330 passed,
  7 skipped. Exact comparisons include 19,680 short HMM boundary cases, 125
  NaN/tie/infinity predecessor combinations, 50,000 DTW paths, random/tied
  sparse/dense inference and six complete HMM fits with training histories.
- Detection and spatial parameters: baseline 275 passed, 8 skipped; final
  focused suite 279 passed, 8 skipped. Exact comparisons include 2,368 static
  window cases, spatial dtype/index/NaN cases and retained autocorrelation lags.
- Trajectories/alignment: baseline and post-edit focused suite 259 passed,
  1 skipped. The final sigmoid eligibility change also passed 21 focused tests.
  Exact comparisons cover 128 AHRS cases, 64 forward/backward cases, 45 extreme
  integration cases including warning lists, 18 Kalman forward cases, and ROI
  intersection with empty/unsorted/overlapping regions.
- Full suite with coverage: 2,747 passed, 41 skipped; core-package coverage 88%.
  Command: `python -m pytest --cov=gaitmap --cov-report=term-missing --cov-report=xml`.
- Repository source lint and all-file formatting passed after the four code
  commits. The obsolete private window-scatter helper was removed after confirming
  it had no remaining consumers, with focused array/static checks repeated.

Targeted experiments compared the actual implementation with functions extracted
from the exact task base. Temporary scripts remain outside the repository.
Warmed component timings demonstrated lower cost for the proposed changes;
they are local measurements, not end-to-end pipeline guarantees. Representative
measurements include DTW path filtering 63.23 to 0.238 ms, HMM boundary matching
47.35 to 2.356 ms, Viterbi 4.685 to 3.441 ms, and ROI intersection 209 to 47.4 ms.
Likelihood-only forward workspace drops from 8 MB to two ten-state rows for
100,000 observations. Emission scores still require their existing storage.

## Integration gates

Run the full pytest suite with coverage, source Ruff and repository formatting.
Resolve every relevant automatic RoboRev review, curate the commit range, then
run the configured five-member `feature_ready` panel against the exact task base.
Keep any final-review corrections as additional commits. Deliver one performance
PR and the separately requested matching correctness PR without merging either.

The four implementation-commit reviews were resolved. The evaluation review
identified the pre-existing matcher defect; its numerical correction is tracked
in the separate PR rather than this result-preserving range. The segmentation,
detection/spatial and trajectory reviews found no actionable issues.
