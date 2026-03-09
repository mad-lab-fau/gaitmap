---
name: tpcp-basics
description: Use when implementing or reviewing core tpcp classes, especially Algorithms and Pipelines, parameter definitions, action methods, result attributes, cloning, and nested parameter handling.
---

# tpcp Basics

Read `../tpcp-builder/SKILL.md` first for the global guardrails.
Also load `../tpcp-datasets/SKILL.md` for custom datasets and `../tpcp-optimization/SKILL.md` for `self_optimize`.

## Build Pattern

- Subclass `Algorithm`, `Pipeline`, or `OptimizablePipeline`.
- Declare any useful class-level parameter annotations.
- In `__init__`, assign each arg directly to `self`.
- Algorithms should accept simple/raw inputs, not whole dataset objects.
- Action methods compute results, store them on `*_` attrs, and return `self`.
- Pipelines consume one dataset datapoint/group, not an entire dataset split.

## Parameters

- In tpcp, all init args are parameters.
- If a value should be tunable/trainable/searchable, expose it in `__init__`.
- Use `set_params(...)` for programmatic updates, including nested updates like `algo__threshold=...`.
- Nested parameter annotations belong on the current class, e.g. `algorithm__threshold: OptimizableParameter[float]`.

## Action Methods

- Custom algorithms should set `_action_methods` to their action name(s), e.g. `"detect"`.
- Pipelines already use `run`/`safe_run`.
- Prefer `@make_action_safe`.
- `safe_run()` checks:
  - returns `self`
  - writes at least one `*_` result
  - does not modify parameters

## Cloning

- Clone before each per-datapoint execution of a nested algorithm.
- Clone before mutating a nested algorithm/object inside `run` or `self_optimize`.
- `clone()` copies parameters but drops results and other non-parameter attrs.
- tpcp clones nested tpcp objects recursively and deep-copies other objects.
- Unlike `sklearn.clone`, tpcp keeps fitted sklearn estimator state because trained models are treated as parameters.

## Mutable Defaults

- Wrap defaults like `list`, `dict`, `np.ndarray`, `pd.DataFrame`, tpcp objects, sklearn estimators, or custom class instances in `cf(...)`.
- For dataclasses/attrs, use their own factories instead of `cf(...)`.

## Common Mistakes

- Doing parameter validation or derived-parameter setup in `__init__`.
- Giving a parameter a trailing `_`; that suffix is reserved for results.
- Forgetting to clone a nested algorithm before calling it.
- Running one algorithm instance repeatedly and expecting older results to remain.
- Storing learned templates/models on ad-hoc attrs instead of init parameters.

## Minimal Pattern

```python
class MyPipe(Pipeline[MyDataset]):
    algo: Parameter[MyAlgo]
    output_: pd.DataFrame

    def __init__(self, algo: MyAlgo = cf(MyAlgo())):
        self.algo = algo

    def run(self, datapoint: MyDataset):
        algo = self.algo.clone()
        algo = algo.detect(datapoint.signal, datapoint.sampling_rate_hz)
        self.output_ = algo.events_
        return self
```

## Source of Truth

- `https://tpcp.readthedocs.io/en/latest/guides/general_concepts.html`
- `https://tpcp.readthedocs.io/en/latest/guides/algorithms_pipelines_datasets.html`
- `https://tpcp.readthedocs.io/en/latest/auto_examples/algorithms/_01_algorithms_qrs_detection.html`
- API docs for `Algorithm`, `Pipeline`, `make_action_safe`, and `clone`
