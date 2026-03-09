# HMM Backend Refactor Investigation

This note captures the first refactor step for the HMM code in `gaitmap_mad`.
The immediate goal is not to replace `pomegranate` yet.
The goal is to separate the public gaitmap/tpcp model API from the current `pomegranate` implementation so that:

- we can keep reproducing the current models and outputs,
- we can remove the Python 3.9 dependency bottleneck later,
- and we can swap the HMM implementation backend in a controlled follow-up step.

## Summary

The current design mixes three concerns in the same objects:

1. topology/configuration of the HMM,
2. training/prediction primitives,
3. backend-native trained model state.

This is the main reason `pomegranate` leaks into the public parameter surface.
Today, `SimpleHmm` and `RothSegmentationHmm` both carry raw `pomegranate` models as init parameters, and the codebase contains custom serialization and clone hacks to keep these objects tpcp-compatible enough for `clone()`, hashing, and JSON export.

The recommended intermediate step is:

- keep exactly one trained `model` parameter on the public HMM algorithm,
- move all submodel configuration into pure config objects,
- introduce a backend object with stateless primitives,
- keep intermediate submodels as results only when needed for debugging.

## Current Surface

The current `pomegranate` dependency is visible in multiple layers:

- `gaitmap/base.py` has special JSON encoding/decoding for `pomegranate.hmm.HiddenMarkovModel`.
- `SimpleHmm` stores a trained `pomegranate` model in `model`.
- `RothSegmentationHmm` stores three trained `pomegranate` models:
  - `stride_model.model`
  - `transition_model.model`
  - `model`
- `packages/gaitmap_mad/.../hmm/_utils.py` contains `_HackyClonableHMMFix` and `_clone_model()` only to make backend-native models clonable and hash-stable enough for tpcp.

This leads to two design problems:

1. `stride_model` and `transition_model` are not configuration objects. They are trainable model holders.
2. The public algorithm surface depends on backend-native model objects instead of backend-neutral model state.

## tpcp Constraints

The replacement needs to stay compatible with the tpcp object model:

- all learned state that must survive `clone()` needs to be an init parameter,
- `self_optimize()` may only update exposed optimizable parameters,
- nested configuration should stay as parameter objects and not become ad-hoc internal state,
- debug-only artifacts should be results with a trailing `_`, not parameters.

For the HMM code this implies:

- `model` must stay an optimizable parameter,
- submodel topology/training options should be pure parameters,
- trained stride/transition submodels should not remain nested optimizable parameters unless they are required for inference.

For `RothSegmentationHmm`, only the final combined model is required for prediction.
Therefore the stride and transition submodels should become temporary training artifacts, not persistent parameters.

## Recommended Public Object Model

### 1. Replace model-like subobjects with config objects

`SimpleHmm` should not remain a trainable wrapper in the new design.
It should become a pure configuration object, for example:

```python
class SimpleHmmConfig(_BaseSerializable):
    n_states: int
    n_gmm_components: int
    architecture: Literal["left-right-strict", "left-right-loose", "fully-connected"]
    algo_train: Literal["viterbi", "baum-welch", "labeled"]
    stop_threshold: float
    max_iterations: int
    name: str
```

This keeps the existing semantics needed to reproduce the current models, but removes the trained backend-native model from the config object.

### 2. Introduce a top-level model definition object

For the Roth model, pass topology/training configuration as a dedicated nested parameter instead of nested trainable submodels:

```python
class RothHmmModelConfig(_BaseSerializable):
    stride: SimpleHmmConfig
    transition: SimpleHmmConfig
    initialization: Literal["labels", "fully-connected"]
```

Then the public `RothSegmentationHmm` surface becomes:

```python
class RothSegmentationHmm(BaseSegmentationHmm):
    model_config: RothHmmModelConfig
    feature_transform: BaseHmmFeatureTransformer
    backend: BaseHmmBackend
    algo_predict: Literal["viterbi", "map"]
    algo_train: Literal["viterbi", "baum-welch"]
    stop_threshold: float
    max_iterations: int
    verbose: bool
    n_jobs: int
    name: str
    model: OptiPara[Optional[HmmModelState]]
    data_columns: OptiPara[Optional[tuple[str, ...]]]
```

This is the main recommended interface change.
It satisfies the requirement that there is only one trained `model` parameter on the public object.

### 3. Keep intermediate models as results, not parameters

If intermediate stride and transition models are still useful for debugging, expose them as:

- `stride_model_`
- `transition_model_`

These are results.
They should not be part of the stable parameter surface.

## Recommended Backend Surface

The backend should be a small stateless object with primitives.
The backend should not know about `SingleSensorData`, stride lists, or feature transforms.
Those remain the job of the gaitmap/tpcp algorithm layer.

The backend should operate on already prepared feature-space arrays and labels.

Recommended minimal surface:

```python
class BaseHmmBackend(_BaseSerializable):
    def initialize_model(
        self,
        *,
        data_sequence: Sequence[np.ndarray],
        labels_sequence: Sequence[Optional[np.ndarray]],
        config: SimpleHmmConfig,
    ) -> HmmModelState: ...

    def fit_model(
        self,
        *,
        model: HmmModelState,
        data_sequence: Sequence[np.ndarray],
        labels_sequence: Sequence[Optional[np.ndarray]],
        algorithm: str,
        stop_threshold: float,
        max_iterations: int,
        verbose: bool,
        n_jobs: int,
    ) -> tuple[HmmModelState, Any]: ...

    def predict_hidden_state_sequence(
        self,
        *,
        model: HmmModelState,
        data: np.ndarray,
        algorithm: Literal["viterbi", "map"],
    ) -> np.ndarray: ...

    def extract_model_definition(self, *, model: HmmModelState) -> HmmModelDefinition: ...

    def build_model(
        self,
        *,
        definition: HmmModelDefinition,
        name: str,
        freeze_distributions: bool = False,
    ) -> HmmModelState: ...
```

### Why this surface

`RothSegmentationHmm.self_optimize_with_info()` currently needs exactly these primitives:

- initialize and fit a stride submodel,
- initialize and fit a transition submodel,
- predict labeled state sequences from trained submodels,
- extract fitted distributions and transition matrices from trained submodels,
- build a combined model from explicit topology plus fitted emissions,
- optionally freeze emissions,
- fit the combined model.

Everything else is glue code and should stay outside the backend.

## Recommended Model State Representation

The trained `model` parameter should not be a raw `pomegranate` object.
It should be a backend-owned serializable wrapper, for example:

```python
class HmmModelState(_BaseSerializable):
    backend_name: str
    payload: dict[str, Any]
```

or, if a stronger type split is preferred:

```python
class BaseHmmModelState(_BaseSerializable):
    pass


class PomegranateHmmModelState(BaseHmmModelState):
    payload: dict[str, Any]
```

The important part is that the public gaitmap object graph only sees `HmmModelState`, not `pomegranate.hmm.HiddenMarkovModel`.

This lets us remove the special `HiddenMarkovModel` handling from `gaitmap/base.py` once the refactor is complete.

## How Topology Should Be Passed

The topology should not be passed as a pre-built backend model.
It should be passed as explicit configuration.

For the immediate refactor, the smallest useful split is:

- `SimpleHmmConfig`
  - state count
  - emission count
  - submodel architecture
  - submodel-local training settings
- `RothHmmModelConfig`
  - stride submodel config
  - transition submodel config
  - combined-model initialization mode

This is preferable to passing raw transition matrices directly because:

- it preserves the current high-level API,
- it remains easy to serialize and compare,
- and it still allows a backend to derive exactly the current model topology.

Direct matrix-based topology objects can be added later if we want a lower-level custom model builder API.

## What Should Stay Outside the Backend

The following should remain in gaitmap-level code:

- feature extraction and inverse transformation,
- converting stride lists into stride/transition training sequences,
- deriving the fully labeled combined training sequence,
- deciding whether the combined model uses `labels` or `fully-connected` initialization,
- validating gaitmap-specific input datatypes and feature column names.

These steps are not backend-specific.
They are part of the algorithm definition.

## Migration Plan

Recommended order:

1. Introduce `BaseHmmBackend`, `SimpleHmmConfig`, `RothHmmModelConfig`, and a backend-owned `HmmModelState`.
2. Add a `PomegranateHmmBackend` that reproduces the current behavior.
3. Refactor `RothSegmentationHmm` to use `model_config + backend + model`.
4. Keep intermediate submodels as results only, if needed.
5. Remove `_HackyClonableHMMFix` from the public HMM classes once `model` no longer stores raw `pomegranate` objects.
6. Remove the `HiddenMarkovModel` special cases from `gaitmap/base.py` after all serialized HMM paths use backend-owned model state objects.
7. Add a second backend only after the public surface is stable and regression tests confirm matching outputs.

## Non-Goals for This Step

This step should not:

- introduce a second HMM backend yet,
- optimize or simplify the Roth training logic,
- change feature extraction,
- change the meaning of the current architecture choices,
- or change the expected model outputs.

The purpose is decoupling, not behavioral change.

## Open Questions

These decisions are still open and should be resolved before implementation:

1. Should `SimpleHmmConfig` keep the current `algo_train` field, or should training strategy move entirely into backend fit calls?
2. Should `HmmModelState` be a single generic wrapper with `backend_name`, or should each backend get its own model-state subclass?
3. Do we want to preserve intermediate stride/transition models as serialized debug artifacts, or are result attributes sufficient?
4. Should the future low-level API expose an explicit `HmmModelDefinition` with transition matrices and emission descriptors, or is that only needed internally by the backend?

## Recommended Decision

For the next implementation step, use this combination:

- `model_config: RothHmmModelConfig`
- `backend: PomegranateHmmBackend`
- `model: HmmModelState`
- `stride_model_` and `transition_model_` as optional result attributes only

That is the narrowest change that removes backend-native objects from the public parameter surface while preserving the current Roth model behavior.
