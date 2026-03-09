# Step 1: Unified Config Surface

## Goal

Replace the dedicated `stride_model` and `transition_model` init parameters with a single composite model config while
still using the current `pomegranate` model objects internally.

## Target API

The HMM class should accept one nested config object that describes:

- the available submodules,
- each submodule's local HMM settings,
- how submodules are connected,
- and how the combined model is trained.

Example target surface:

```python
class HmmSubModelConfig(_BaseSerializable):
    n_states: int
    n_gmm_components: int
    architecture: Literal["left-right-strict", "left-right-loose", "fully-connected"]
    algo_train: Literal["viterbi", "baum-welch", "labeled"]
    stop_threshold: float
    max_iterations: int
    name: str


class HmmConnectionConfig(_BaseSerializable):
    from_module: str
    from_state: int
    to_module: str
    to_state: int
    initial_probability: float | None = None


class CompositeHmmConfig(_BaseSerializable):
    modules: dict[str, HmmSubModelConfig]
    connections: tuple[HmmConnectionConfig, ...]
    initialization: Literal["labels", "fully-connected"]
    combined_algo_train: Literal["viterbi", "baum-welch", "labeled"]
    combined_stop_threshold: float
    combined_max_iterations: int
    name: str
```

## Scope

This step should only change configuration flow.
It should not yet change:

- `RothSegmentationHmm.model`
- the stored pretrained model format
- the use of raw `pomegranate` models internally
- `_HackyClonableHMMFix`
- `gaitmap/base.py` HMM serialization

## Implementation Notes

- Keep the current Roth behavior by providing a default two-module config with `transition` and `stride`.
- Move the current submodel-specific parameters into default config values.
- Keep the current training loop structure:
  - train each configured submodule independently
  - combine them by building a final `pomegranate` model
  - run the final refinement pass on the combined model
- The training/data-splitting logic can still be Roth-specific in this step.
  The point is to stabilize the input surface first.

## Compatibility Strategy

- Backwards compatibility for init parameters is optional.
- If needed, provide a thin compatibility shim that maps legacy `stride_model` / `transition_model` inputs to the new
  config internally.
- The stored model format should remain unchanged in this step so that the pretrained artifact continues to load
  without migration work.

## Expected Outcome

After this step:

- the public constructor uses a single config object,
- custom multi-module setups become expressible,
- but the trained model is still a `pomegranate` model and behavior should be unchanged.
