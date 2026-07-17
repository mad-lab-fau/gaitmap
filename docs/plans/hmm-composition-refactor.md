# Composable HMM refactor

Status: implementation in progress

## Motivation

The existing HMM implementation couples three independent concerns:

- the Roth stride-segmentation training schedule;
- the definition and fitting of a single HMM;
- pomegranate's runtime model representation.

It also treats the two Roth roles, `stride` and `transition`, as fixed HMM
concepts. This prevents callers from composing and independently training
additional models such as walking strides, stair strides, running strides, and
their connecting regions.

The refactor will make HMM topology and fitted weights backend-neutral, extract
generic model composition, and express Roth as a preset and annotation adapter
over generic multi-model orchestration.

## Decisions

### One model abstraction

`HmmModel` represents both lifecycle states:

- **untrained**: complete topology and emission definitions, without fitted
  probabilities or distributions;
- **fitted**: the same topology and stable state identities plus
  backend-neutral fitted parameters.

Topology and fitted parameters remain separate internally, but callers do not
coordinate separate `HmmTopology` and `HmmModel` objects. Fitting returns the
same abstraction with its optimizable fitted-parameter value populated.

`SimpleHmm` will not remain as a second public model abstraction.

### Stable state identity

Topology constructors assign deterministic state identities. Composition
namespaces them using their part names, for example:

```text
("walking", "state_0")
("stairs", "state_0")
("locomotion", "running", "state_3")
```

Inference adapters may compile these identities to integer array positions,
but fitting and flattening must not change their public identity. Every named
part automatically forms a state group.

### Generic composition

`HmmModel.compose()` accepts any number of atomic or composed models and a
part-level route graph. The generic model has no knowledge of strides,
transitions, activities, or gait events.

Every model exposes an entry distribution, internal transition matrix, and exit
distribution. A fitted macro route from model `A` to model `B` expands as:

```text
P(A_i -> B_j) = exit_A[i] * route(A, B) * entry_B[j]
```

For untrained models the same expansion creates an allowed-transition mask.
Composition is deterministic in-process computation and does not belong to a
numerical backend.

The initial interface supports entry-to-exit routing. Arbitrary state-level
graphs remain possible through `HmmModel.from_matrix()`.

### Atomic trainer

The training adapter fits exactly one HMM. Its interface supports the two
required fitting modes:

```python
trainer.fit(model, observations, labels, train="all")
trainer.fit(model, observations, labels, train="transitions")
```

`"all"` fits emissions and transition/start/end probabilities.
`"transitions"` preserves fitted emissions bit-for-bit and only fits the
transition, start, and end probabilities.

The modern pomegranate adapter supports both modes. SciPy remains inference
only. No backend owns topology construction, composition, region extraction, or
Roth orchestration.

### Generic composite training

`CompositeHmm` owns the reusable multi-model training schedule. Training input
is a sequence of complete, non-overlapping region tables:

```text
start  end   model
0      80    transition
80     310   walking
310    370   transition
370    610   stairs
```

The `model` value names a part in the composed topology. Generic orchestration
does not silently interpret gaps as transitions or another background class.

Training performs these steps:

1. Transform the observations and labelled regions into feature space.
2. Extract the observation sequences belonging to each model part.
3. Generate topology-compatible initialization labels for each region.
4. Ask the atomic trainer to fit every part independently.
5. Decode each labelled region with its fitted part.
6. Stitch those namespaced predictions into complete hidden-state sequences.
7. Compose the fitted parts using the declared route graph.
8. Ask the atomic trainer to fit the combined model with
   `train="transitions"`.

Region tables must be sorted, contained in the recording, non-overlapping, and
complete over the samples used for combined fitting. Labels must name existing
model parts. All composed parts must use the same feature columns.

### Roth is an adapter and preset

`RothSegmentationHmm` provides:

- the standard Roth feature transform;
- the default stride/transition model topology;
- conversion of stride-border annotations into a complete region table by
  assigning annotated strides to `"stride"` and gaps to `"transition"`;
- the semantic declaration that the `"stride"` state group represents stride
  candidates.

It delegates generic fitting and prediction to the same model and backend
interfaces used by configurations with more than two parts.

### Segmentation remains generic

`HmmStrideSegmentation` consumes the segmentation-HMM interface rather than a
concrete Roth class. It can be configured with one or more segment groups. When
multiple groups are selected, output stride lists retain the originating group
as a type label.

### Serialization and legacy loading

Newly trained models serialize topology, stable identities, named groups,
composition metadata, feature schema, and NumPy fitted parameters. Serialized
models contain no torch or pomegranate objects.

`RothSegmentationHmm.from_legacy_json()` remains a one-way adapter for untouched
pomegranate 0.14 exports. It reconstructs the Roth feature configuration and
state groups and compiles the exact fused parameters for inference. The bundled
historical JSON must remain byte-for-byte unchanged and pass through this same
adapter. Every loaded fitted model must work with every inference backend
available on the running Python version.

## Intended interface

```python
walking = HmmModel.left_right(
    n_states=20,
    n_gmm_components=6,
    cycle=False,
    starts="first",
    ends="last",
)
transition = HmmModel.left_right(
    n_states=5,
    n_gmm_components=3,
    cycle=True,
    starts="all",
    ends="all",
)

topology = HmmModel.compose(
    parts={
        "transition": transition,
        "walking": walking,
        "stairs": HmmModel.left_right(20, n_gmm_components=6),
        "running": HmmModel.left_right(16, n_gmm_components=4),
    },
    routes={
        "transition": ("walking", "stairs", "running"),
        "walking": ("transition",),
        "stairs": ("transition",),
        "running": ("transition",),
    },
    starts=("transition",),
    ends=("transition",),
)

trained = CompositeHmm(model=topology).self_optimize(
    data_sequence,
    region_list_sequence,
    sampling_rate_hz=204.8,
)
```

The precise container types may be adjusted to follow tpcp composite-parameter
conventions, but the observable behavior and separation of responsibilities are
fixed by this plan.

## TDD implementation slices

- [x] `HmmModel` represents a validated unfitted left-right topology and becomes
      fitted without changing state identity or topology.
- [x] Matrix construction rejects invalid dimensions, unreachable states, and
      emission definitions that do not cover every state.
- [x] N-way composition deterministically namespaces states, creates groups,
      and expands entry-to-exit routes.
- [x] Composition of fitted models preserves emissions and produces normalized
      backend-neutral combined probabilities.
- [x] Atomic pomegranate training accepts an `HmmModel` and implements
      `train="all"`.
- [x] Transition-only fitting preserves emission arrays exactly.
- [x] SciPy and modern pomegranate decode the same fitted model abstraction.
- [x] `CompositeHmm` trains three or more named parts from complete labelled
      regions and produces a fitted combined model.
- [x] Invalid, incomplete, overlapping, or unknown-labelled region tables fail
      with actionable errors.
- [x] `RothSegmentationHmm` converts stride borders into the generic region
      contract and follows the same independent-fit/compose/final-fit path.
- [x] Custom Roth topologies derive state counts and stride groups from their
      model definition rather than constants.
- [x] Untouched historical JSON loads through the public legacy adapter and
      reproduces exact SciPy hidden states and stride borders.
- [x] Legacy-loaded and newly trained models run through all supported inference
      adapters.
- [x] `HmmStrideSegmentation` supports a generic segmentation-HMM and typed
      output for multiple segment groups.
- [x] Examples and generated interface documentation cover pretrained
      inference, legacy custom files, custom Roth training, and multi-part
      composite training.
- [x] Delete the flat hard-coded 5/20-state training path and all superseded
      model/conversion code.

Each item is implemented as an individual red-green-refactor slice. Tests target
the public interface and observable results rather than private composition or
backend helpers.

## Acceptance criteria

- The original bundled JSON has no diff from the pre-refactor repository.
- Custom historical JSON files load without modification.
- SciPy inference remains exactly compatible with legacy hidden states and
  stride lists.
- State counts, GMM component counts, internal topology, and part routing are
  model parameters rather than Roth constants.
- A single atomic trainer implementation is reused for submodel and final-model
  fitting.
- A configuration with at least walking, stair, running, and transition parts
  can be trained without writing a custom orchestration algorithm.
- New fitted models contain no backend runtime objects and can switch inference
  adapters after loading.
- Python 3.9 supports SciPy and optional legacy pomegranate inference; modern
  pomegranate training and inference retain their explicit newer-Python
  boundary.
- All HMM, package, lint, formatting, documentation, and supported-version test
  suites pass.

## Non-goals

- Reproducing modern pomegranate training results bit-for-bit with pomegranate
  0.14.
- General arbitrary partial-parameter fitting beyond `"all"` and
  `"transitions"`.
- Automatically inferring semantic labels for unannotated training gaps in the
  generic composite module.
- Recursive training of nested composites from hierarchical annotations in the
  first implementation. Nested topology composition and stable state paths must
  not prevent adding it later.
