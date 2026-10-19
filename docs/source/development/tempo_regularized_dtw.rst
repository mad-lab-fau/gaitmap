.. _tempo_regularized_dtw:

Experimental tempo-regularized subsequence alignment
====================================================

``ExperimentalTempoBarthDtw`` is a separate experimental segmentation algorithm.
It leaves ``BaseDtw``, ``BarthDtw`` and ``ConstrainedBarthDtw`` unchanged.
No improvement in gait accuracy has been established. The defaults are
uncalibrated, and existing DTW thresholds do not transfer to its normalized score.

The method adapts the expanded-state second-order warp regularization discussed
by Deriso and Boyd in `section 6.2 of A General Optimization Framework for Dynamic
Time Warping <https://arxiv.org/html/1905.12893v2#S6.SS2>`_. Their framework motivates
retaining additional state. The log-tempo penalty, free-start block model and bin
integration here are our adaptation, not a published gait-specific method or a
claim that an existing package implements this exact design.

Discretized model
-----------------

Template X has m samples at rate ft; signal Y has n samples at rate fs.
Samples represent constant, half-open time bins. Template duration T = m / ft
includes the last sample bin, rather than using the last sample timestamp as its
endpoint. Divide template time into K equal blocks, each of duration dt = T / K.
Blocks may cut sample bins; no template sample is discarded or averaged away.

Signal block edges b[0], ..., b[K] are integers with strictly positive advances
r[k] = b[k+1] - b[k]. Each template block maps linearly to its signal block.
The initial signal edge is free. Actual local tempo is::

    v[k] = r[k] / (fs * dt)       # signal seconds per template second

Values above one correspond to slower, longer matches. Every positive integer
advance realizing a tempo within the finite positive bounds becomes a state.
At most 64 states are allowed. An empty or oversized alphabet raises an error;
there is no silent pruning, clipping or beam search. Increase K or narrow the
bounds to reduce the alphabet. Bound-to-integer conversion tolerates only eight
floating-point ulps of arithmetic roundoff. Inspect ``tempo_alphabet_`` or the
numerical result's ``advances`` and ``tempos`` to see the realized discretization.

A one-block match and every feasible constant-speed match have zero tempo-change
penalty. No cost favors unit initial tempo. There are no zero/infinite slopes.

Loss, normalization and units
-----------------------------

For block k starting at signal edge b and consuming r signal bins, let L(k,b,r)
be the integral of squared feature mismatch over its template-time interval,
divided by T. Both signals are piecewise constant, so the implementation computes
this integral exactly for the defined sample-bin model. It intersects all
sample-bin boundaries, using integer arithmetic for the intersection geometry.
Every consumed signal bin and template bin has positive weight. A long transition
cannot omit a mismatching signal feature in its interior.

Every block contributes total data weight 1 / K, regardless of its tempo.
The objective is::

    data_cost  = sum over all weighted sample pairs of weight * sum_features((X - Y)^2)
    tempo_cost = penalty * sum_(k=1..K-1)((log(v[k]) - log(v[k-1]))^2)
    J          = data_cost + tempo_cost
    reported segmentation cost = sqrt(J)

Data loss is a mean over template time, without division by the number of
features or by the matched signal duration. Feature scaling therefore matters.
The numerical alignment reports squared costs; ``costs_`` and ``cost_function_``
report square roots. The accumulated projection contains squared costs.

The penalty coefficient has squared transformed-feature units. There is no
hidden grid factor. Changing K changes both the allowed tempos and the
regularization objective, so reconsider the coefficient. To approximate a
continuous penalty ``alpha / T * integral((d(log(v))/dt)^2 dt)``, use the discrete
coefficient ``alpha / (T * dt)``. Here alpha has squared-feature times seconds
squared units. This describes the grid scaling; the shipped default directly
uses the discrete sum above.

At zero coefficient the solver finds the optimum of this same block lattice.
It does not become ordinary unconstrained DTW, whose path alphabet and
unnormalized cell-sum objective differ.

Dynamic program and duration feasibility
----------------------------------------

State D(k,j,q) retains the cheapest arrival after k blocks, at exclusive signal
edge j, with last tempo index q and integer advance r[q]. Impossible states have
infinite cost. For all allowed starts, initialize without an initial tempo cost::

    D(1,j,q) = L(0, j-r[q], r[q])

    D(k+1,j,q) = L(k, j-r[q], r[q])
                + min_p(D(k,j-r[q],p)
                        + penalty * (log(v[q])-log(v[p]))^2)

The solver retains all tempo hypotheses, including a more expensive arrival that
may win after a later continuation. It never filters one tempo onto the cheapest
projected cell. Exactness refers to this discretized model, with ordinary
floating-point rounding, rather than to a continuous warp optimum.

Ties prefer lower cost, then earlier initial start, then ascending predecessor
tempo index. Endpoint ties use ascending final tempo index after cost and start.
This can select longer constant matches on flat data; no unit-tempo preference
is added to resolve such ambiguity.

Optional numerical duration limits are inclusive integer counts of consumed
signal bins. Class duration limits in seconds remain strict, as in existing
Barth postprocessing. They convert to floor(min_seconds * fs) + 1 and
ceil(max_seconds * fs) - 1 samples. A duration-valid unrestricted winner is
already an exact feasible optimum. Otherwise the solver recomputes that endpoint
with every allowed initial edge in the feasible start interval. Since the
endpoint is fixed, those starts have identical future validity and can safely
compete in the same DP. Simply discarding a duration-invalid cheapest start
would miss feasible alternatives and is not used here.

Snapping and Barth conflict handling occur after numerical matching. They can
change or reject intervals; original paths, costs and weighted metadata always
describe the unsnapped matches. As with the existing endpoint helpers, peak
selection is a heuristic on the optimum endpoint curve, not global optimization
of a nonoverlapping stride sequence.

Memory, time and reconstruction
-------------------------------

The solver rolls ``min(max_advance,n) + 1`` signal columns of K-by-Q float costs
and integer start IDs. It retains a K-by-n cost projection and endpoint metadata,
never a K-by-n-by-Q floating-point history. Requested paths are deliberately
recomputed from the selected start in a window bounded by K * max_advance,
using compact uint8 predecessor tempo IDs.

Let B be the largest number of overlapping sample pairs in one block/tempo and
L the maximum match duration in signal samples. A free-start pass costs
``O(n*K*Q*(Q+B))``. Each duration fallback or requested traceback costs at most
``O(L*K*Q*(Q+B))``, where L is bounded by K * max_advance. In the adversarial case
all duration-feasible endpoints need fallback, giving
``O(n*L*K*Q*(Q+B))`` work. The fallback saves state memory at a real runtime cost.
``duration_fallback_count_`` exposes that cost's trigger.

Rolling numerical memory is ``O(K*Q*min(max_advance,n))``, plus the 2D
``O(K*n)`` output. A requested path temporarily needs ``O(K*L*Q)`` byte parents
and another 2D local projection. Returned alignments also retain their paths and
positive template-time weights. Weights sum to one and reproduce the data term;
block edges and tempos reproduce the regularizer. An unweighted cell sum does
not reproduce this objective.

Interfaces and example
----------------------

The class is available from ``gaitmap.stride_segmentation`` and
``gaitmap_mad.stride_segmentation``. The numerical functions and result types are
exported from ``gaitmap_mad.stride_segmentation.dtw``::

    result = tempo_regularized_subsequence_dtw(
        template_array, signal_array,
        template_sampling_rate_hz=100, signal_sampling_rate_hz=100,
        n_segments=20, tempo_bounds=(0.5, 2.0), tempo_change_penalty=0.1,
    )
    alignment = tempo_dtw_paths(template_array, signal_array, result, [endpoint])[0]

Numerical endpoints are inclusive sample indices. ``alignment.block_edges``
are exclusive bin edges, so its interval is ``[block_edges[0], block_edges[-1])``.
Class ``acc_cost_mat_`` has block rows, not template-sample rows; its interior
rows are unrestricted partial optima and its final row enforces durations.
Unequal physical rates are supported without resampling. A template rate is
required. With resampling enabled, the model uses the resampled template bins.

The public class inherits directly from BaseStrideSegmentation. An ephemeral
private Barth adapter reuses input dispatch and stride postprocessing without
changing any existing solver or base class. This avoids inherited run-limit
parameters that do not describe the new model, including tpcp's warning for
subclasses that omit parent parameters. The public object passes action safety,
cloning, real joblib caching and JSON parameter roundtrips with warnings treated
as errors.

Run ``examples/stride_segmentation/experimental_tempo_dtw_stride_segmentation.py``
for a small executable comparison and bounded single-worker measurements. The
comparison changes block edges from ``[0,1,3,5,6]`` at zero penalty to
``[0,1,2,4,6]`` at coefficient 2. Data loss increases from 1.5 to 1.875 while the
selected path uses fewer abrupt tempo changes. This demonstrates the objective,
not an improvement on gait data.

Measured bounded run
--------------------

A Linux single-worker run on 2026-10-19 used a 100-sample template at 100 Hz,
K = 10, 16 tempo states, and 500/1,000 signal samples. BLAS/OpenMP thread counts
were fixed to one. Timings include Python setup for the numerical call and
allocation, after the example had loaded the numerical kernel. They are warm
runtimes, not cold compilation or cold-process startup measurements.

* 500 samples, unrestricted: 0.00413 s, zero fallbacks.
* 500 samples, duration exactly 191: 0.23309 s, 310 fallbacks.
* 1,000 samples, unrestricted: 0.00799 s, zero fallbacks.
* 1,000 samples, duration exactly 191: 0.60655 s, 810 fallbacks.

The all-zero input makes a constant-tempo unrestricted optimum win every tie.
No constant integer advance over ten blocks has duration 191, so **every
endpoint eligible for that duration triggers a fallback**. This exercises the
worst-case fallback frequency; it is not a full-cohort performance result.

Whole-process peak resident memory was **260.27 MiB**, measured from Linux
``/proc/self/status`` VmHWM. This includes imports, Numba/cache loading, the toy
segmentation and all measured cases. The 1,000-sample retained cost projection
was only 80,000 bytes; it is not a substitute for peak-memory measurement.
In an earlier invocation, raw ``getrusage`` reported 824.92 MiB even at
interpreter startup, due to inherited pre-exec process history. The example
reports that raw value separately and uses process-image VmHWM on Linux to avoid
attributing inherited memory to the solver. No retained-matrix mocks were used.
