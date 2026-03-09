---
name: tpcp-builder
description: Use when building, reviewing, or refactoring code that subclasses tpcp Dataset, Algorithm, Pipeline, or uses tpcp optimization/validation. Gives the global rules, main pitfalls, and points to focused tpcp skills for basics, datasets, and optimization.
---

# tpcp Builder

Read this first, then load the focused sibling skill(s) you need:

- Basics: `../tpcp-basics/SKILL.md`
- Datasets: `../tpcp-datasets/SKILL.md`
- Optimization: `../tpcp-optimization/SKILL.md`

## Mental Model

- `Dataset`: index + lazy access to actual data.
- `Algorithm`: reusable step with one or more action methods.
- `Pipeline`: glue code that runs on exactly one dataset datapoint/group.
- Optimization in tpcp means "data-driven changes to init parameters", including model training.

## Non-Negotiable Rules

- Every `__init__` argument must be stored unchanged on `self` under the same name.
- Do not validate, coerce, derive, or mutate parameters in `__init__`; do that in `run`/action methods.
- Do not use `*args` in tpcp object `__init__`.
- Parameter names must not contain `__` or end with `_`.
- Any mutable default or nested object default must use `cf(...)` or a dataclass/attrs factory.
- Results live on attributes ending with `_`.
- Action methods and `self_optimize` must return `self` (or `(self, info)` for `self_optimize_with_info`).
- Algorithms should take the simplest raw inputs they need; pipelines are the place that consume dataset datapoints.
- `run`/action methods must not modify parameters.
- Clone nested algorithms/objects before running or mutating them.
- Never optimize on test data.
- Any value that training/search changes must be an exposed init parameter.

## Highest-Risk Pitfalls

- Shared mutable defaults create silent cross-instance state and can cause train-test leakage.
- Reusing one nested algorithm instance across datapoints overwrites results and leaks fitted state.
- Building a non-deterministic dataset index breaks splits, caching, and reproducibility.
- Passing a multi-row/multi-group dataset into `Pipeline.run` violates the intended interface.
- Storing learned state outside init parameters makes `clone()` drop it and breaks optimization semantics.
- Marking `PureParameter` incorrectly can invalidate optimization shortcuts; default to plain `Parameter` unless sure.

## Safe Defaults

- Prefer `pipeline.safe_run(datapoint)` over `pipeline.run(datapoint)`.
- Prefer `@make_action_safe` on custom action methods.
- Prefer `@make_optimize_safe` and `Optimize(...)` for `self_optimize` pipelines.
- Prefer `GridSearch`/`GridSearchCV`/`OptunaSearch` for brute-force or black-box search.

## Quick Triage

If behavior is strange, check these first:

1. Mutable default or shared nested object?
2. Missing `clone()` before nested `run`/`detect`/`self_optimize`?
3. Dataset `create_index()` deterministic and sorted?
4. `run` touching parameters instead of only writing `*_` results?
5. `self_optimize` changing non-optimizable params or non-parameter attrs?

## Source of Truth

- Concepts: `https://tpcp.readthedocs.io/en/latest/guides/general_concepts.html`
- Dataset/algorithm/pipeline model: `https://tpcp.readthedocs.io/en/latest/guides/algorithms_pipelines_datasets.html`
- Evaluation and leakage rules: `https://tpcp.readthedocs.io/en/latest/guides/algorithm_evaluation.html`
- Optimization rules: `https://tpcp.readthedocs.io/en/latest/guides/optimization.html`
