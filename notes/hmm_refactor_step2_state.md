# Step 2: Introduce Generic HMMState

## Goal

Introduce a backend-neutral serializable `HMMState` and change `RothSegmentationHmm.model` to store this state instead
of a raw `pomegranate` model.

## Core Idea

The model state should support:

- one compiled flat HMM for inference,
- optional hierarchical/composite structure,
- backend provenance for compatibility warnings,
- and fully explicit emission and transition parameters.

## Target Shape

The state should not be hardcoded to Roth's current `transition` and `stride` modules.
It must allow arbitrary named submodules.

Suggested split:

```python
class BackendInfo(_BaseSerializable):
    backend_id: str
    backend_version: str | None = None
    state_schema_version: int = 1


class HmmGraphState(_BaseSerializable):
    start_probs: np.ndarray
    end_probs: np.ndarray
    transition_probs: np.ndarray


class GaussianEmissionState(_BaseSerializable):
    kind: Literal["gaussian"]
    mean: np.ndarray
    covariance: np.ndarray
    covariance_type: Literal["full", "diag", "sphere"]
    frozen: bool = False


class GaussianMixtureEmissionState(_BaseSerializable):
    kind: Literal["gaussian_mixture"]
    weights: np.ndarray
    components: tuple[GaussianEmissionState, ...]
    frozen: bool = False


class FlatHmmState(_BaseSerializable):
    graph: HmmGraphState
    emissions: tuple[GaussianEmissionState | GaussianMixtureEmissionState, ...]
    state_names: tuple[str, ...] | None = None


class CompositionEdge(_BaseSerializable):
    from_module: str
    from_state: int
    to_module: str
    to_state: int
    weight: float | None = None


class CompositeHmmState(_BaseSerializable):
    submodels: dict[str, FlatHmmState]
    combined: FlatHmmState
    cross_module_edges: tuple[CompositionEdge, ...]
    metadata: dict[str, Any] | None = None


class HmmModelState(_BaseSerializable):
    trained_with: BackendInfo
    compiled: FlatHmmState
    composite: CompositeHmmState | None = None
```

## Scope

This step should change:

- `RothSegmentationHmm.model`
- the serialization format for trained HMMs
- loading of the existing pretrained model via migration into `HmmModelState`

This step should not yet require a full backend abstraction.
The current implementation can still directly use `pomegranate` internally and only convert at the boundary where the
public parameter is read or written.

## Migration Work

- Add a loader that converts the existing pretrained JSON artifact into `HmmModelState`.
- Preserve enough information to warn when a state was produced by the legacy backend.
- Update tests so cloning, hashing, and JSON roundtrips work without raw `pomegranate` objects in public parameters.

## Expected Outcome

After this step:

- the public trained-model parameter is backend-neutral,
- legacy pretrained artifacts still load through migration,
- and the public API no longer depends on `pomegranate` object serialization.
