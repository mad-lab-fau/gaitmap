# Step 3: Add Pomegranate 0.14 Backend Abstraction

## Goal

Introduce a dedicated `pomegranate 0.14` backend that is responsible for converting between backend-native models and
the generic `HMMState`, and for executing flat-model training/inference primitives.

## Backend Responsibility

The backend should own:

- initialization of a flat HMM from config and initial labels,
- fitting a flat model,
- prediction of hidden state sequences,
- conversion from backend-native model to `FlatHmmState`,
- conversion from `FlatHmmState` to backend-native model.

The algorithm layer should still own:

- feature transformation,
- extraction of training sequences,
- composition of multiple named submodules,
- derivation of cross-module transitions,
- orchestration of the final combined-model refinement pass.

## Suggested Surface

```python
class BaseHmmBackend(_BaseSerializable):
    def initialize(self, config, data, labels) -> FlatHmmState: ...
    def fit(self, model, data, labels, *, fit_mode) -> tuple[FlatHmmState, Any]: ...
    def predict(self, model, data, *, algorithm) -> np.ndarray: ...
    def to_backend_model(self, model: FlatHmmState) -> Any: ...
    def from_backend_model(self, model: Any) -> FlatHmmState: ...
```

For `pomegranate 0.14`, `fit_mode` likely needs at least:

- `"full"`
- `"transitions_only"`

## Scope

This step should:

- move the remaining direct `pomegranate` calls out of the public algorithm classes,
- isolate legacy backend details such as cloning quirks and state-name quirks,
- and establish the adapter pattern needed for later backends.

This is the step where `_HackyClonableHMMFix` should start disappearing from the public HMM classes, because the
public parameter should already be `HmmModelState` from step 2.

## Follow-Up

Once this step is complete, future work becomes much simpler:

- add a `pomegranate 1.x` backend if desired,
- add a SciPy inference-only backend using the compiled flat state,
- and add compatibility warnings based on `HmmModelState.trained_with`.

## Expected Outcome

After this step:

- the current legacy implementation is isolated behind a backend adapter,
- the public HMM API is config-driven and backend-neutral,
- and future backend replacements no longer require another public API redesign.
