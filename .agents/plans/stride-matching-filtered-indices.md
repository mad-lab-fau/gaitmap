# Stride matching filtered-index correction

## Accepted scope

- Deliver a separate bug-fix PR to `master`, based exactly on
  `10afaccc2d2e74dba239168c38e0be1a0877edc0`.
- Own the narrow matching correction, public regression test, changelog and this
  task context. Include no performance commits and do not alter performance PR100.
- Correct the final Chebyshev check to use original left row IDs from
  `valid_matches`, rather than positions within that filtered array.
- Preserve mutual Manhattan nearest-neighbor selection and existing tie criteria.
  Public signatures and output schemas stay the same. This is a correction to the
  documented inclusive per-boundary tolerance, not a new matching algorithm.
- Prove red then green through the agreed public `match_stride_lists` seam.
  For left `[[0, 10], [20, 30], [40, 50]]` and right
  `[[21, 31], [41, 51]]`, tolerance 1 and one-to-one matching, retain left/right
  pairs `(1, 0)` and `(2, 1)` and leave only left stride 0 unmatched.
- Use the existing environment with this checkout first on `PYTHONPATH`. Run
  focused evaluation checks, not broad algorithm tests. Do not investigate Mobgap.
- Commit one coherent fix with its regression and release note, push, and create
  and register the PR after that commit. Resolve automatic RoboRev feedback and
  the required `feature_ready` panel over the exact full range. Do not merge.

## Verification before the implementation commit

Environment: `/home/arne/.t3/worktrees/gaitmap/t3code-cdc00b1b/.venv/bin/python`,
with `PYTHONPATH=$PWD:$PWD/packages/gaitmap_mad/src`.

- Baseline: `python -m pytest tests/test_evaluation_utlis/test_stride_segmentation.py -q`
  passed all 49 existing tests at the exact base.
- Red: the same command with `-k tolerance_after_unmatched_stride` failed before
  the source correction. It returned five unmatched rows instead of the expected
  three rows containing the two valid pairs.
- The source fix only maps filtered candidate positions back to original left
  rows for the final Chebyshev check.
- Green: `python -m pytest tests/test_evaluation_utlis -q --disable-warnings`
  passed 120 tests, including the new public regression and existing event,
  segmentation, and scoring tests. It reported 74 numerical warnings in the
  parameter-error tests. No broad algorithm suites were run.
- `ruff check gaitmap/evaluation_utils/stride_segmentation.py`, `ruff format --check`
  on the changed source and test, and `git diff --check` passed. An additional
  direct lint of the test file reported 47 docstring/exception-style diagnostics;
  CI's configured lint target is the two package source trees, not tests. The new
  test follows the existing test style.
- RoboRev v0.71.0, daemon health, post-commit hook and local configuration were
  checked before implementation and were healthy.
