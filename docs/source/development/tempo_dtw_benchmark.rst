.. _tempo_dtw_benchmark:

Tempo DTW: held-out accuracy and computational cost
===================================================

The :ref:`experimental tempo model <tempo_regularized_dtw>` improves held-out
stride detection on this healthy-participant cohort compared with the previously
optimized, corrected constrained DTW. The improvement also occurs with the
new model's tempo-change penalty disabled. This experiment therefore supports
the new alignment formulation, but does not establish a benefit from its
smoothness penalty. Current inference is slower and uses less peak memory.

Protocol and model selection
----------------------------

This uses the gaitmap-bench Sensor Position Comparison instep challenge 1.0.0,
source ``50b8d08584ca50f7d10c539d4070978f40e62214``: all 14 participants, both
instep sensors, full recordings at 204.8 Hz, and 12,171 expert reference strides.
The official scorer matches both borders within 30 ms, truncated to six samples,
with one-to-one matching. Feet use explicit left/right sensor names. Prepared
input hashes and dataset/package provenance are retained in the manifest.

The five participant KFold partitions are unchanged. For each outer fold,
configuration selection maximizes pooled training-participant F1 on only its
11 or 12 training participants. The remaining participants are evaluated once
with the selected configuration. The fixed original Barth template is never
fitted to these labels. Data-only numerical computations may be shared across
folds; objective aggregation uses each fold's training IDs exclusively.

The new search has eight numerical settings, each evaluated at seven thresholds,
for 56 configuration/threshold pairs per fold:

* Template-time blocks: 10 or 20.
* Penalty coefficients at 20 blocks: 0, 0.01, 0.1 or 1. At 10 blocks the actual
  coefficient is halved, preserving the stated continuous-penalty grid scaling.
* Normalized score thresholds: 0.1, 0.15, 0.2, 0.25, 0.3, 0.4 or 0.6.

All settings use ``gyr_ml`` alone, the axes selected in every fold by the prior
constrained-DTW search. Physical tempo bounds are 0.65 to 2.0 signal seconds
per template second. Existing Barth duration limits are 0.6 to 3 seconds and
border snapping is 300 ms. The actual integer tempo alphabet is retained for
every numerical setting; changing block count also changes that alphabet.
The normalized score threshold is not comparable with an old DTW threshold.

The main search is allowed to choose zero penalty. A separately tuned control
fixes the penalty to zero and selects block count/threshold on the same training
participants. It isolates the incremental contribution of the smoothness prior;
it still uses the new positive-tempo block lattice and weighted sample-bin loss.

The comparator is the completed corrected constrained-DTW search with 100 seeded
Optuna trials per fold, not the library defaults in the earlier
:ref:`fixed-parameter comparison <constrained_dtw_benchmark>`. Its winning axes
were also ``gyr_ml`` in all folds. The two algorithms have different search
spaces and budgets, so this is a comparison of the specified tuned procedures.
Baseline numerical source hashes are checked unchanged. Its saved predictions
are rescored with the current official scorer and the identical references used
for both new arms; old metric tables are not copied.

Accuracy
--------

.. list-table:: Pooled held-out detection over all 14 participants
   :header-rows: 1

   * - Model
     - Precision
     - Recall
     - F1
     - TP
     - FP
     - FN
   * - Corrected constrained DTW
     - 97.2098%
     - 96.7546%
     - 96.9817%
     - 11,776
     - 338
     - 395
   * - Tempo model, penalty selectable
     - 98.4568%
     - 99.0716%
     - 98.7632%
     - 12,058
     - 189
     - 113
   * - Same block model, penalty fixed to zero
     - 98.3803%
     - 99.3098%
     - 98.8429%
     - 12,087
     - 199
     - 84

Compared with corrected constrained DTW, the main search gains 282 true
positives and removes 149 false positives. Its pooled F1 difference is
+1.7815 percentage points, with participant-paired bootstrap 95% percentile
interval [1.0533, 2.9494] points. The zero-penalty control gains 311 true positives
and removes 139 false positives, with F1 difference +1.8612 points and interval
[1.0415, 3.2040]. Intervals resample the 14 participants 2,000 times with seed 93;
they describe this cohort, not independent clinical validation.

The main search selects zero penalty in four folds and a small positive penalty
in fold 3. Its held-out F1 is 0.0797 percentage points below the zero-penalty
control; the paired interval [-0.2533, 0.0163] includes zero. The positive-penalty
choice loses 29 true detections and removes ten false positives relative to the
control. This provides no evidence that slope smoothness improves accuracy here.
It does not isolate which of the other formulation changes produced the gain.

.. list-table:: Main search selections, using training labels only
   :header-rows: 1

   * - Fold
     - Blocks
     - Actual penalty
     - Threshold
     - Training F1
   * - 0
     - 10
     - 0
     - 0.25
     - 98.8488%
   * - 1
     - 10
     - 0
     - 0.25
     - 98.9357%
   * - 2
     - 20
     - 0
     - 0.25
     - 98.7616%
   * - 3
     - 10
     - 0.005
     - 0.25
     - 98.9755%
   * - 4
     - 10
     - 0
     - 0.25
     - 98.8230%

The zero-penalty control selects the same block counts and thresholds, with
penalty zero in every fold. Participant-macro F1 is 96.9556% for corrected
constrained DTW, 98.7683% for the main search and 98.8445% for the control.

Controlled inference time and memory
------------------------------------

Performance probes run sequentially in fresh Linux processes with BLAS, OpenMP
and Numba numerical thread limits set to one. Each performs a complete untimed
warmup on the same recording, followed by two uncached public ``segment()`` calls
with fresh algorithm instances. Loading prepared inputs, constructing the
algorithm and scoring are excluded from the warm timings. Template processing,
numerical alignment, path reconstruction, snapping and conflict handling are
included. Timed predictions must equal the stored held-out predictions.

The preregistered core subset is first/middle/last participant: ``4d91``, ``80b8``
and ``e54d``. Before inspecting held-out accuracy, the first test participant of
every fold whose training selection used positive penalty was added. This adds
``9b4b`` and includes an actual smoothness-prior configuration. All three arms
are timed for all four participants, with randomized arm order for the core
subset. These timing/RSS results cover four recordings per foot, rather than
all 14 accuracy participants. Two repeats give a practical comparison, not a
high-precision performance study.

.. list-table:: Warm full-recording inference, both feet per participant
   :header-rows: 1

   * - Model
     - Mean of participant medians, seconds
     - Time / constrained
     - Median process peak RSS, MiB
     - Median retained cost projection, MiB
   * - Corrected constrained DTW
     - 3.786
     - 1.00x
     - 1326.8
     - 558.1
   * - Tempo model, penalty selectable
     - 11.043
     - 2.92x
     - 381.6
     - 27.9
   * - Same block model, penalty fixed to zero
     - 11.134
     - 2.94x
     - 382.1
     - 27.9

The time ratio divides the sum of four participant medians by the corresponding
baseline sum. Whole-process RSS is Linux ``VmHWM`` and includes interpreter,
imports, prepared data, full warmup and both timed calls. It is not incremental
kernel memory. The retained projection column is a separate array-size measure,
not a substitute for peak RSS. All arms run through the same benchmark imports.

This measures the current implementation, which deliberately recomputes paths
instead of retaining global state predecessors. That reduces retained state at
a real runtime cost. The solver still carries tempo states at zero penalty;
a specialized zero-penalty solver and more efficient traceback are possible
future improvements. The measured slowdown is not an intrinsic lower bound of
tempo regularization.

No real-recording search or timing call required duration fallback. At the
200-sample resampled template length, the lower tempo bound already implies a
numerical match longer than 0.6 seconds, and the upper bound is below 3 seconds.
This benchmark therefore does not exercise adversarial duration recomputation.
The :ref:`bounded synthetic example <tempo_regularized_dtw>` separately records
that worst-case trigger: roughly 0.01 seconds unrestricted versus 0.61 seconds
for a 1,000-sample duration-constrained case. Later snapping can still cause
ordinary postprocessing rejection.

Search cost and reproducibility
--------------------------------

The search required 112 unique numerical computations: eight settings for each
of 14 participants. Every setting checked all seven thresholds through the
public pipeline, for 784 threshold-equivalence checks. Compact caches retain
candidate intervals and costs, not DP matrices. Selected held-out predictions
for both new arms were recalculated without patches or numerical caches.
All 784 count rows, ten training-only selections, 42 matching tables and their
12,171 reference counts per arm were independently audited.

The eight-worker candidate-generation phase took 287.846 seconds of elapsed
service time and 2,141.324 CPU seconds, with a roughly 2.1 GiB group peak and
zero swap. Resumed scoring and fresh held-out validation took
82.129 seconds. Together
these recorded compute phases took approximately
6.17 minutes,
excluding the review pause and the subsequent sequential timing experiment.
Sum of parallel job calculation times was 2079.628
seconds; threshold verification added 14.436
aggregate job seconds. Summed job wall times are not CPU time or inference
latency. Search throughput must not be used as a latency comparison.

A review correction changed baseline rescoring after numerical generation.
The generation manifest is retained alongside the current manifest, with an
explicit source-change note. Numerical functions, candidate construction,
threshold checks, inputs and search grid were unchanged; all 112 caches were
preserved. The later timing-subset extension changes coverage, not model fitting.

From this feature checkout, use the same dataset preparation as
``scripts/optimize_constrained_dtw.py`` and its completed baseline results::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
      PYTHONPATH="$PWD:$PWD/packages/gaitmap_mad/src" .venv/bin/python \
      scripts/benchmark_tempo_dtw.py \
      --bench-repo /path/to/gaitmap-bench \
      --prepared-dir /path/to/optimized-rerun/prepared \
      --baseline-dir /path/to/optimized-rerun \
      --workers 8 --repeats 2

Use ``--stage search`` or ``--stage timing`` to run the phases separately. The
full runner includes the positive-penalty timing-subset extension automatically.
The baseline directory contains its manifest, fold selections and all participant
``*-heldout-strides.joblib`` predictions. No raw sensor recordings are included
in the report artifacts.

Download the :download:`summary <../../data/tempo_dtw_benchmark/summary.json>`,
:download:`participant metrics <../../data/tempo_dtw_benchmark/heldout-per-participant.csv>`,
:download:`timing records <../../data/tempo_dtw_benchmark/timing.json>`,
:download:`search counts <../../data/tempo_dtw_benchmark/search-counts.csv>`,
:download:`fold selections <../../data/tempo_dtw_benchmark/selections.json>`,
:download:`baseline selections <../../data/tempo_dtw_benchmark/baseline-results.json>` and
:download:`provenance <../../data/tempo_dtw_benchmark/manifest.json>`.
:download:`All 42 raw matching tables <../../data/tempo_dtw_benchmark/matching-tables.zip>`
are provided as a compressed archive.
