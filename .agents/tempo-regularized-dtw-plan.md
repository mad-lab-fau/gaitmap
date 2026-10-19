# Experimental tempo-regularized subsequence DTW implementation plan

COMPACTION CONTINUITY: Re-read implement-code-change and the task-defining artifacts before continuing after compaction or session restoration.

## Ownership and current gate

- Task brief: the implementation-owner request in this conversation, including its design requirements and acceptance evidence.
- Checkout: `t3/tempo-regularized-subsequence-dtw`, initially clean at `7bc4f9e2`.
- Parent owns integration, final branch review, push, and PR delivery. Do not push, widen PR 96, merge, rebase finalized history, or spawn agents.
- Preserve existing DTW classes, public parameters, result formats, and numerical behavior. The scoped compatibility policy is `.agents/refactor-policy.md`; the explicit task authorizes a separate new experimental algorithm.
- **Confirmed:** user answered "Yes, both interfaces". Numerical aligner and public `segment()` seams are agreed; implementation can proceed.
- Applicable skills read: implement-code-change, TDD, working-with-roborev, reviewable-commits and its during-implementation reference, backwards-compatibility, unslop.
- RoboRev CLI and daemon work; shared repository post-commit hook invokes RoboRev. No open reviews in this checkout at task start.

## Proposed numerical model, before implementation

This proposal is an adaptation of second-order warp regularization, not a published gait-specific algorithm. Primary reference read: Deriso and Boyd, *A General Optimization Framework for Dynamic Time Warping*, section 6.2, https://arxiv.org/html/1905.12893v2#S6.SS2. That section expands the state for a higher-order warp penalty; the particular log-tempo penalty and block model below are our design.

### Time, positions and tempo

Let template samples be X[0:m] at rate ft and signal samples Y[0:n] at rate fs. Both are vector-valued. Treat each sample as a constant on its half-open time bin. Thus the template duration is T = m / ft, including the last sample bin. This convention is deliberate and differs from treating the sample timestamps alone as an interval of length (m - 1) / ft.

Divide template time into K equal blocks of duration dt = T / K. Block edges may cut template bins; do not drop or resample those bins. Signal edges b[k] are integers, 0 <= b[0] < ... < b[K] <= n. A block consumes r[k] = b[k+1] - b[k] signal bins and maps its template-time interval linearly onto those bins. The signal start b[0] is free.

The finite ordered tempo alphabet is given by all positive integer advances r satisfying:

    min_tempo <= r / (fs * dt) <= max_tempo

The actual local tempo is v[r] = r / (fs * dt), measured in signal seconds per template second. Larger v means a longer, slower stride. Bounds must be finite and strictly positive. Expose the realized alphabet, since rounding changes it. Reject an empty alphabet or an alphabet larger than a documented small cap, rather than silently pruning or clipping it. Initial proposal: cap 64 states, configurable K via `n_segments`, default K = 20. Tiny examples explicitly choose smaller K.

There are no zero/infinite-slope transitions. There is no preferred initial tempo and no penalty for the first block. Any constant feasible tempo has zero tempo-change cost, whether fast or slow.

### Data cost and objective

Let phi map template time to signal time by the selected linear block maps. Define each block's data cost as the exact integral for the piecewise constant input model:

    L(k, b, r) = integral over template block k of ||X(t) - Y(phi(t))||^2 dt / T

Compute it by intersecting template sample-bin edges with inverse-mapped signal sample-bin edges. Every consumed template bin and signal bin contributes its positive overlap. A long advance cannot jump over a mismatching signal feature. Do not replace a block with an endpoint-only or block-average comparison.

Every block has total data weight 1 / K, independently of tempo. The complete data term is a mean over template time of the sum of squared feature differences, without normalization by feature count or matched signal duration.

    J = sum_k L(k, b[k], r[k])
        + tempo_change_penalty * sum_(k=1..K-1) (log(v[k]) - log(v[k-1]))^2

The reported scalar detection cost is sqrt(J). Report the data term and regularization term separately in numerical alignments. The penalty coefficient has squared transformed-feature units. The sum is defined on this fixed K grid, with no hidden time-step factor. Changing K changes the regularizer and requires reconsidering its coefficient. To approximate a continuous penalty alpha / T times the integral of squared time derivative of log tempo, set the discrete coefficient to alpha / (T * dt). Do not claim invariance under arbitrary grid changes.

At zero penalty this is exact unregularized alignment on the same positive-tempo block lattice. It is not ordinary unconstrained DTW and cannot reproduce its zero/infinite-slope paths.

### State, recurrence, initialization and ties

Retain competing states D(k, j, q), where k blocks have been consumed, j is the exclusive signal edge and q is the last advance/tempo index. A state's block start is j - r[q].

    D(1, j, q) = L(0, j - r[q], r[q])

for every allowed initial start, with no initial regularization.

    D(k+1, j, q) = L(k, j - r[q], r[q])
                   + min_p [D(k, j - r[q], p)
                            + penalty * (log(v[q]) - log(v[p]))^2]

Impossible states have infinite cost. No single filtered tempo is attached to a projected cell. Exactness claims apply only to this lattice and specified block loss. No beam search, tempo pruning, or iterative refinement is planned.

Use deterministic comparisons: cost first, earliest initial start second, then ascending predecessor tempo index. At an endpoint, use cost, earliest start, then ascending final tempo index. Strict deterministic tie behavior must agree between accumulation and traceback recomputation.

### Duration constraints without invalid start merging

The optional numerical min/max durations are inclusive integer counts of consumed signal bins. Class duration parameters in seconds convert to counts consistently with the existing strict duration filters. Snapping is a later postprocessing operation and can still invalidate a final interval; it does not redefine the numerical optimum.

A single retained start per tempo is sufficient for the unrestricted objective, but not for duration-constrained optima. First compute unrestricted endpoint optima with start IDs. An unrestricted winner satisfying the duration interval is already an exact feasible optimum. For every invalid endpoint winner, recompute a bounded local DP with that endpoint fixed and **all** allowed initial edges in [end - max_duration, end - min_duration]. Their future validity is then identical because the endpoint is fixed. No start is discarded on the basis of a cheapest path with a different future validity. Replace the endpoint's score/start/terminal tempo with the feasible optimum or infinity.

Only starts within K * max_advance of an endpoint can be feasible. Clip the local window using that bound and recording boundaries. This fallback has a worse runtime when many unconstrained winners fail duration limits; document and measure it. Do not replace it with an approximate post-filter or claim duration-optimal matches after simply rejecting an unrestricted winner.

### Memory and path recovery

Iterate over signal edges with an explicit ring of min(max_advance, n) + 1 columns. Each ring column holds K by Q float state costs and integer start IDs. Keep an optional 2D K-by-n projection for inspection; never allocate K-by-n-by-Q floating costs. The projected column for exclusive signal edge j is j - 1, consistent with sample-indexed detection curves. Interior rows describe unrestricted partial-block optima; optional duration feasibility applies to the complete-template final row only.

Record final endpoint starts and tempo IDs. Deliberately recompute only requested paths in their bounded windows with compact predecessor tempo IDs. A fixed selected start and endpoint bounds this window by K * max_advance. Recompute using the same recurrence and tie ordering. Free global predecessors are unnecessary. This trades extra work per selected match for bounded memory.

Recover all positive-overlap sample pairs plus their weights from the selected block edges. The numerical alignment contains:

- integer sample-pair path, with first pair (0, start) and last pair (m - 1, end - 1);
- per-pair template-time weights, summing to one for a complete path;
- K + 1 exclusive signal block edges and realized K tempos;
- separate weighted data cost, tempo-change cost, and total squared objective.

The same integer pair can be aggregated if needed; no consumed interval is omitted. A path's score is the weighted pair loss plus its block-tempo penalty. Plain unweighted DTW cell sums are not this objective.

## Proposed public interface and integration

- New numerical module `_tempo_dtw.py`, explicitly exported through `gaitmap_mad.stride_segmentation.dtw`. Proposed entry point `tempo_regularized_subsequence_dtw(template, signal, *, template_sampling_rate_hz, signal_sampling_rate_hz, n_segments=20, tempo_bounds=(0.5, 2.0), tempo_change_penalty=..., min_duration_samples=None, max_duration_samples=None) -> TempoDtwResult`.
- Result exposes endpoint costs, optional 2D block-cost projection, realized advances/tempos, and selected endpoint start/state metadata. Companion `tempo_dtw_paths(template, signal, result, endpoints)` returns `TempoDtwAlignment` objects. Freeze exact naming/default penalty when the first agreed test is written; no arbitrary numeric calibration claim.
- New `ExperimentalTempoBarthDtw` sibling of BarthDtw, inheriting directly from BaseStrideSegmentation and composing an ephemeral private Barth adapter for extraction/scaling, multi-sensor dispatch, matching and postprocessing. The adapter overrides `_segment_single_dataset` to handle numerical results explicitly. This removes the tpcp warning caused by omitting irrelevant inherited run-limit parameters. Do not pretend they are ConstrainedDtwResult or recover paths from the 2D projected matrix.
- Avoid editing the existing BaseDtw pipeline. A small explicit subclass pipeline is preferable here to expanding its constrained-result type branches or storing numerical scratch state on the algorithm.
- Constructor follows tpcp conventions, including CloneFactory for template defaults. Expose the new tempo/grid parameters and applicable template/matching/postprocessing options. Omit old consecutive-run constraints from the new class constructor. Defaults should disable snapping and duration filters until those features are specifically verified and documented.
- Existing results keep their familiar types in the new class, but its documented `acc_cost_mat_` has block rows, not template-sample rows. Its final row has normalized squared total costs. Add aligned path weights/block edges/tempos so callers can reproduce the objective.
- Export the class through MAD dtw/stride_segmentation and gaitmap's lazy MAD bridge. Add its API entry in the actual existing stride-segmentation API page after locating it.
- For unequal rates with resampling disabled, use physical rates directly; if the template rate is missing, require it rather than silently assume a physical tempo. With resampling enabled, use the rate of the resulting template.
- New development documentation and one small executable synthetic example. State that gait accuracy has not been evaluated. No full experiment search and no changes to the existing experiment harness.

## Acceptance criteria and planned review units

1. Solver plus meaningful numerical verification, once that seam is confirmed.
   - Independent tiny exhaustive oracle enumerates whole start/advance sequences and scores whole warp intervals with an independently written integration method, not the production recurrence or block helper.
   - Tiny exhaustive comparisons include duration constraints and optima requiring alternate initial starts.
   - Construct two arrivals at one position with different tempos whose ranking reverses after a common continuation. Reject a cheapest-cell-only implementation.
   - Uniform feasible fast and slow alignments both have zero regularizer and correct endpoints.
   - Hidden mismatching samples inside large advances contribute loss.
   - A sudden-tempo-change shortcut exhibits the penalty's effect, without a gait-accuracy claim.
   - Zero coefficient behavior matches this lattice's exhaustive oracle, not ordinary DTW.
   - Deterministic ties, endpoint/path/weight/tempo/score agreement, and unreachable endpoints.
   - Verification-and-commit gate: focused agreed-seam tests and scoped lint/format pass; inspect staged diff; create coherent local solver commit and inspect its asynchronous RoboRev job.
2. Segmentation, docs and executable measured example integration.
   - If segment() seam confirmed, prove end-to-end costs/intervals, single/multi-sensor dispatch, cache/clone behavior where materially relevant, and preserved stride-list formats.
   - Run relevant existing-DTW regressions. Do not create implementation-detail mocks or retaining side-effect mocks.
   - Execute synthetic example in one bounded worker process. Record cold/warm timing distinctions and real process peak RSS using resource.getrusage or /usr/bin/time. Include a duration-fallback case. Exclude plots/imports/JIT from a warm timing only when stated; report the actual process peak including them.
   - Verification-and-commit gate: focused integration and regression checks, scoped lint/format, actual runnable example and memory/runtime record; coherent local integration commit.
3. Delegated handoff.
   - Read finish-implementation-stack when implementation is actually complete; honor parent ownership rather than repeating integration/push/PR work.
   - Inspect and resolve relevant per-commit RoboRev reviews and report their status. Preserve reviewed history as instructed; no push or integration rewrite.
   - Report hashes, owned files, exact recurrence/interface, checks, measured memory/runtime, discretization and runtime limits, and unresolved decisions.

## Verification record

- Initial tree: clean; HEAD `7bc4f9e2`; no task code changes.
- RoboRev: `roborev status` healthy; shared post-commit hook present; `roborev list --status running --status queued --status done --open` found no jobs.
- Existing-DTW baseline passed: **214 passed, 1 skipped, 33 warnings in 3.50 s**. Command: `PYTHONPATH="$PWD:$PWD/packages/gaitmap_mad/src" /home/arne/Documents/repos/github/gaitmap/.venv/bin/python -m pytest tests/test_stride_segmentation/test_constrained_dtw_optimality.py tests/test_stride_segmentation/test_base_dtw.py tests/test_stride_segmentation/test_barth_dtw.py tests/test_stride_segmentation/test_constrained_barth_dtw.py -q`. The explicit PYTHONPATH selects this checkout instead of the shared environment's original editable install. Existing warnings concern caching serialization, template rates, deprecated APIs, overlaps, and float-index slicing.
- `git diff --check` passed after adding this plan. Only the new plan is untracked; no tracked files changed.
- Both test seams confirmed on resumption. Initial red/green slice begins with exact identity alignment and path metadata.

## Solver slice verification

- Confirmed seams: numerical interface and segment(), explicitly approved by user.
- Red: identity test failed on missing public numerical exports. Green: identity and weighted metadata passed after solver addition.
- Red: duration-oracle test failed because duration arguments were absent. Green: exact fixed-endpoint feasible-start fallback added. Oracle helper required converting NumPy integers to Python integers for Fraction hashing.
- Final solver gate: 65 numerical tests passed in 0.43 s; scoped Ruff check passed and files formatted. Tests cover independent whole-warp rational integration, ranking reversal, uniform tempo, penalty tradeoff, consumed samples, rates, duration alternatives, unreachable endpoints, and stable ties.
- Implementation adjustment: no global predecessor history; per-request bounded recomputation uses uint8 parents. Public defaults: 20 blocks and penalty 0.1, explicitly uncalibrated.
- No existing DTW base class was edited. No push or PR work.

## Integration slice verification and design resolution

- Solver commit 98cbc2a8; dedicated numerical roundoff correction 6d5f1c2e. Review 15771 found integer advance-boundary roundoff; fixed with stable rate arithmetic, eight-ulp boundary tolerance and a red/green seven-bin regression. Review 15773 passed the correction. Both jobs closed.
- Public class uses composition with a private ephemeral Barth adapter, not inheritance from BarthDtw. This avoids adding irrelevant run-limit parameters and removes tpcp's omitted-parent-parameter warning. The adapter is never cloned/cached; its settings are explicit pure-solver cache arguments. Existing base classes are unchanged.
- Confirmed segment() tests cover nonzero regularized objective, duration-optimal matches, both peak/threshold endpoint methods, recording boundaries, multisensor data, real memory-mapped joblib caching, clone/JSON/action safety, empty results, unequal rates, resampling and snapping with original metadata preserved.
- New integration tests pass with UserWarnings treated as errors (6 passed). Final combined numerical/integration/existing-DTW check is recorded below after completion.
- Scoped Sphinx HTML build with autodoc/numpydoc for the new model and APIs passed with warnings treated as errors. Output: /tmp/tempo-dtw-docs-11jjwzd7/html. The first build caught a short RST title underline, which was corrected. This is not a whole-repository docs build.
- Executable example ran in one fresh process, OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1. Final warm timings: n=500 unrestricted 0.00413 s / duration 191 0.23309 s (310 fallbacks); n=1000 unrestricted 0.00799 s / duration 191 0.60655 s (810 fallbacks). Every duration-feasible endpoint falls back in the adversarial cases.
- Actual Linux process-image peak RSS (VmHWM): 260.27 MiB, including imports/Numba/example/all cases. n=1000 output cost projection: 80,000 bytes. Linux getrusage can retain inherited pre-exec peaks: an earlier interpreter already reported 824.92 MiB before imports while VmHWM was 14.66 MiB, so the example reports both and uses VmHWM. Final run raw getrusage was 258.16 MiB.
- No cohort search, experiment harness changes, existing-solver modifications, push, PR edits or child agents.
- Parent still owns final integrated-branch review and delivery. Only our unfinalized task-local commits may be curated; do not rewrite any history at or before base 7bc4f9e2.

- Final integration gate: 286 passed, 1 skipped, 33 pre-existing warnings in 1.87 s across both new test modules and the four existing DTW modules. Scoped Ruff check/format and git diff --check passed. All public constructor parameters documented.

## Integration review corrections

- Review 15778 reproduced two empty-candidate bugs: empty threshold indices had float dtype, and snapping an empty path list reached a reduction on empty indices. Added red regressions for both endpoint methods with snapping, normalized endpoint dtype, and bypassed path reconstruction/postprocessing for empty candidates. New integration suite: 8 passed with UserWarnings treated as errors.
- Guarded optional Unix resource import so Windows can run/import the example without it. Unsupported platforms report null RSS rather than fail. Real Windows was not available; a bounded smoke run blocks resource imports and exercises the win32 guard.
- Corrected the example's gallery label/title syntax. Focused Sphinx HTML now also includes extracted example documentation and passes with warnings as errors.

## Parent integration and benchmark extension

Subagent handoff finalized local solver839fc9a9 and integrationfcc285c1; parent audited both coherent commits, no unresolved fixups, base7bc4f9e2 preserved. Existing numerical/public-test seam confirmation remains valid. Global/repository configuration has no feature_ready panel; a single whole-feature review will cover the final implementation plus requested benchmark changes once complete, rather than repeatedly reviewing the algorithm-only range.

User extended acceptance: run gaitmap-bench with a small justified hyperparameter search for the new algorithm, report held-out accuracy AND computational performance penalty versus corrected constrained DTW. Preserve 14participants, explicit feet,5outerfolds, official6samplematching. Profile realrecording first, choose bounded small search over score threshold/tempo penalty and optionally discretization. Fixaxes to prior winninggyr_ml to focusbudget. Separate warm inference/wholeprocesspeakRSS from search overhead, use controlled sequential repeated timing and no persistent numericalcache. Include zero-regularization control to attribute effect to new prior rather than blockmodel. No model selection on held-out labels. Record solver hashes, actual parameters/temporalphabet, search configuration and scoring tables. Use a separate draftPR based on parent branch, preserving PR96.

### Parent verification and benchmark protocol

- Full test suite on fcc285c1:2494passed,40skipped,300pre-existingwarnings in91.88s. This covers the integrated library tree; later benchmark-only script/docs edits do not invalidate library result.
- DraftPR97 created via gh-stack as separate layer abovePR96, bothregisteredwithT3; GitHubstack98. Algorithmcommits839fc9a9,fcc285c1 pushed withoutrewritingparenthistory.
- Fixed preregistered numerical grid: K10/20; reference penalty atK20 in[0,.01,.1,1], scaledlinearlybyK/20. Thresholds[.1,.15,.2,.25,.3,.4,.6];56combinations/fold. axesgyr_ml fixed,physicaltempobounds(.65,2), min/maxduration(.6,3)s,snapping300ms. Lower tempo bound yieldsduration>.6s for the200sample/204.8Hz template, avoiding the expensive exact-durationfallback while preserving thepublicduration/peaksettings. Report actual alphabet and anyfallbackcounts. Zero-penalty arm separatelytunesgrid/threshold tocontrolforblockmodelchanges. Main arm canselectzero; noforcedregularization.
- Same14participants/bothfeet/fullrecordings,5outerfolds aspriorcorrectedsearch. RankonlytrainingpooledF1; rerunselectedheldoutpipelinesfresh. PreviousoptimizedcorrectedConstrainedBarthDtw accuracyreusedonlyafterverifyingunchangedsolverhashes; fresh timingpredictionsmustmatchitsstoredheldoutstrides.
- Cacheonlycompactcandidates. Everyconfigurationchecksatall7thresholds throughpublicsegmentwith plaincallable replayofsameDP/paths. NoMagicMockcycles. Finalheldoutrecomputation has no patches/caches. Pilot8configurations on5000samplesperfootpassesall56thresholdchecks;4.622swall,198.1MiBservicepeak,0swap. Ruffcheck/format/diffpass.
- Real54a9pilot:175155samplesleftfoot,warmnumericalpass2.7516s; publicfullsegmentation24.7024s with1349candidates,0durationfallback,356.5MiBservicepeak. Tracebackismajoroverhead. Pilotusedsignalonly,nolabels.
- Performance subset preregistered:first/middle/lastparticipant4d91,80b8,e54d. Forall3arms, useisolatedsequentialfreshprocesses,fullrecordingwarmup then2uncachedtimedsegmentcallswithfreshinstances, numericalthreads1, randomizedarmorderseed93. Reportactualcalls, warmmedians,wholeprocessVmHWM,andretainedprojectionbytes. Segmentationincludespublictemplateprocessing/postprocessing;prepareddata/scoringexcluded. Searchwalltime/totalcompute/verificationoverheadreportedseparately. No throughput-as-latencycomparison.

Benchmarkreview15782 foundcomparatorprovenancegap: runnerreusedoldcountswithoutenforcingsamescorerrevision. Correctedbyrescoringall14savedbaselinepredictionswithcurrentofficialscorerandpreparedreferences, savingall14newbaseline matchingtables. Directcurrent-scorercheck confirmsall14tp/fp/fncounts matchpriorrun. This makesall3armsusesamescoring evenifscorerrevisionchanges. Preserved112/112compactcandidatejobs onstoppingsearchatgeneration/scoringboundary; candidategeneration/solver/grid/inputs unchanged by correction. Manifestgeneration snapshotretained, currentrunnersourcehashupdatedwithexplicitmigrationnote; no cachednumericsinvalidated. Library tree unchanged, fullsuiteevidencevalid. Reviewcorrectionwillremainfixupuntilfinalowned-featurecuration; neverrewritebase7bc4f9e2orPR96.

Training-onlyselectionresult:4/5foldsmainarmselectzeropenalty;fold3selectK10,referencepenalty.01(actual.005),threshold.25. Allthresholdsselected.25;K10exceptfold2K20. Toensureinferencecostofactuallyusedsmoothingisrepresented, supplementpreregistered3participanttimingswithfirstheldoutparticipantofeachfoldwhoseTRAININGselectionusespositivepenalty. That adds9b4b, beforeinspectingheldoutaccuracy. Noadditionalaccuracytuning;original3participantmeasurementsretained. These extra3probeprocessescanrunwithsamepublic--stageprobeCLIafteroriginalservicecompletes; no numerical/script/gridchange.

### Final benchmark audit and report gate

All112candidatejobs/784publicthresholdchecks completed. All28selectednew/controlheldoutpipelines recomputeduncached andmatchedcompactpredictions. All14baselinepredictionsrescoredcurrentofficialscorer; all42rawtablesand784countrowsindependentlyaudited, eacharmcovers14people/12171references. All10foldselectionsre-derivedfromtrainingonly. Bothbenchmarkservicesandsupplementaltimingexitedsuccessfully,0swap. Fourparticipants×3arms×fullwarmup+2timedcallscompleted, matchingstoredheldoutpredictions. Archive42CSVmembersCRCcheckedandbytecomparedtosource.

HeldoutpooledF1:constrained96.9816759%, main98.7632075%,zero98.8428671%. Main+1.78153ppvsbaseline,pairedparticipantbootstrap95%[1.05325,2.94939];mainvszero−.07966pp,CI[−.25330,.01633]. Fourfoldsselectzero;fold3selectactualpenalty.005. Thisdoesnotestablishsmoothnessbenefit.

Warmmeanof4participantmedians:baseline3.78601s,main11.04297s(2.91678x),zero11.13362s(2.94073x). MedianprocessVmHWM:1326.84375/381.58594/382.07422MiB;projectionmedian558.08258/27.90413/27.90413MiB. Everyrealnumericalcallhas0durationfallback;prototypepathrecomputationcausesruntimeoverhead,notintrinsiclowerboundofprior. Generation287.846selapsed/2141.324CPU s,observed~2.1GiBgrouppeak;resumedscoring/validation82.12939s,~6.17minutescomputephasesexcludingreviewpause/timing. Allmetrics/provenance/parameterselections andrawmatchingarchiveindocs/data/tempo_dtw_benchmark;reportdocs/source/development/tempo_dtw_benchmark.rst.

ScopedRuffcheck/formatanddiffpass. StrictfocusedSphinxbuildincludingnewmodel,benchmark,APIandexamplepasses;baselinereportlabelstubusedbecauseunchangedexistingreportalreadyverifiedandfulldocsHMMissuepre-exists. Newdownloadlinkbaseline-resultscheckedinfinalfocusedbuild. Full2494passed40skipped libraryevidencereusedasnoalgorithm/librarycodechangesafterverifiedfcc285c1;laterchangesonlybenchmarkscript/docs/data. Nextgate:reportcommit,itsautomaticreview,ownedfeaturecurationabsorb14634a21into80b629c1withbackupandlease;thenONEfinalfeature review frombase7bc4f9e2. Noprior-finalizedhistoryrewrite.
