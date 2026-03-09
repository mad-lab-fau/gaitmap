---
name: tpcp-optimization
description: Use when implementing or reviewing tpcp self_optimize logic, parameter annotations, Optimize/GridSearch/GridSearchCV usage, and validation workflows that must avoid train-test leakage.
---

# tpcp Optimization

Read `../tpcp-builder/SKILL.md` first for the global guardrails.
Also load `../tpcp-basics/SKILL.md` for cloning/parameter rules and `../tpcp-datasets/SKILL.md` for split semantics.

## Pick the Right Tool

- Use `self_optimize` only for algorithm-specific training logic.
- Do not re-implement brute-force search inside `self_optimize`.
- Use `Optimize(pipeline)` for pipelines that implement `self_optimize`.
- Use `GridSearch` for brute-force search without inner CV.
- Use `GridSearchCV` when hyperparameter search itself needs CV.
- Use `DummyOptimize` if you need a non-optimizable baseline on the same CV folds.

## Parameter Semantics

- `OptimizableParameter`: changed by `self_optimize`.
- `HyperParameter`: changes how `self_optimize` behaves but is not changed by it.
- `PureParameter`: does not affect `self_optimize`; only use when you are certain.
- If unsure, prefer plain `Parameter` over `PureParameter`.

## Hard Rules for `self_optimize`

- Return `self`, or `(self, info)` from `self_optimize_with_info`.
- Modify only parameters marked as optimizable on the current class.
- Do not store learned state on non-parameter attrs.
- Any learned model/template/weights must survive `clone()`, so they must be represented as parameters.
- Clone nested algorithms before training/mutating them.
- Prefer `@make_optimize_safe`; `Optimize(...)` also applies equivalent checks.

## Evaluation Rules

- Never search/tune/train on the final test data.
- Outer evaluation measures the whole training procedure, not one already-trained instance.
- `cross_validate(...)` expects an optimizer object, not a bare pipeline.
- For grouped or stratified splits, create explicit group/label arrays or use `DatasetSplitter`.
- In custom scorers, call `pipeline.safe_run(datapoint)`.

## Common Mistakes

- Putting black-box parameter search into `self_optimize` instead of `GridSearch`/`OptunaSearch`.
- Forgetting to annotate optimizable params, causing safety checks to fail.
- Changing non-optimizable params during `self_optimize`.
- Marking a parameter as `PureParameter` even though it affects training.
- Training a nested algorithm in place and then reusing it across folds/datapoints.
- Calling `self_optimize` directly in user-facing code instead of using `Optimize(...)`.

## Minimal Pattern

```python
class MyPipeline(OptimizablePipeline[MyDataset]):
    model: Parameter[MyAlgo]
    model__weights: OptimizableParameter[np.ndarray]

    def __init__(self, model: MyAlgo = cf(MyAlgo())):
        self.model = model

    def self_optimize(self, dataset: MyDataset, **kwargs):
        model = self.model.clone()
        self.model = model.self_optimize(...)
        return self
```

## Source of Truth

- `https://tpcp.readthedocs.io/en/latest/guides/optimization.html`
- `https://tpcp.readthedocs.io/en/latest/guides/algorithm_evaluation.html`
- `https://tpcp.readthedocs.io/en/latest/guides/algorithm_validation_tpcp.html`
- `https://tpcp.readthedocs.io/en/latest/auto_examples/parameter_optimization/_02_optimizable_pipelines.html`
- `https://tpcp.readthedocs.io/en/latest/auto_examples/parameter_optimization/_03_gridsearch_cv.html`
- API docs for `Optimize`, `GridSearch`, `GridSearchCV`, and `make_optimize_safe`
