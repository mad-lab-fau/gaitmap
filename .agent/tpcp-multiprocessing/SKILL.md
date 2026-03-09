---
name: tpcp-multiprocessing
description: Use when working on tpcp code that uses n_jobs, joblib parallelism, tpcp.parallel, caching in parallel workers, or when debugging multiprocessing, serialization, or global state issues in tpcp.
---

# tpcp Multiprocessing

Read the official `Multiprocessing Caveats` guide first:
`https://tpcp.readthedocs.io/en/latest/guides/multiprocessing_caveats.html`

## Use this skill when

- `validate`, `cross_validate`, `Scorer`, or an optimizer uses `n_jobs`
- global config seems missing in workers
- runtime monkey-patching/decorators/caches behave differently in parallel
- joblib raises pickle or `__main__`-related errors
- heavy imports make parallel runs unexpectedly slow

## Main Caveats

- Worker processes do not automatically inherit runtime global state changes from the parent process.
- Joblib `loky` workers are reused, so worker-side global mutations can leak into later jobs.
- Serialization is often the real failure point, not the parallel API itself.
- Objects defined in `__main__`, lambdas, nested classes/functions, and runtime-replaced globals are high risk.
- Heavy optional imports can dominate worker startup cost.

## tpcp-Specific Guidance

- For global state restoration, use `tpcp.parallel.delayed` together with `register_global_parallel_callback(...)`.
- Assume runtime-applied decorators or caches are not visible in workers unless explicitly restored there.
- If tests require a clean worker pool, shut down the reusable loky executor explicitly.
- If debugging cost outweighs the speedup, fall back to `n_jobs=1`.

## Fast Triage

1. Missing config only in workers: global-state problem, use `tpcp.parallel`.
2. Error mentions `__main__`: move code to an importable module.
3. Works once, fails later: suspect process-pool reuse and leaked worker state.
4. Parallel is slower than serial: inspect import cost and serialization overhead.

## Source of Truth

- `https://tpcp.readthedocs.io/en/latest/guides/multiprocessing_caveats.html`
- API docs for `tpcp.parallel`
- GitHub issue `#119`
