---
name: tpcp-datasets
description: Use when implementing or reviewing tpcp Dataset classes, dataset indexes, grouping, subsetting, data accessors, and split/group label behavior for validation workflows.
---

# tpcp Datasets

Read `../tpcp-builder/SKILL.md` first for global guardrails.
Also load `../tpcp-basics/SKILL.md` when the dataset feeds a custom pipeline.

## Required Shape

- Subclass `Dataset[...]`.
- `__init__` must include `groupby_cols=None, subset_index=None` at the end of the signature and forward both to `super().__init__(...)`.
- Implement `create_index()` to return the full metadata index as a `pd.DataFrame`.
- Keep actual file/data loading out of `create_index()`; load lazily in properties/methods.

## Index Rules

- `create_index()` must be deterministic. tpcp calls it twice and will error if outputs differ.
- Sort the final index explicitly; do not rely on filesystem order or `set` iteration.
- Index columns should be valid Python identifiers. Invalid names break ergonomics around `get_subset`, `group_label`, and `group_labels`.
- If you use a typed named-tuple group label generic, its field names and order must match index columns exactly.

## Data Accessors

- Properties that expose actual data should usually require a single row via `assert_is_single(...)`.
- Properties that expose group-level data should require a single current group via `assert_is_single_group(...)`.
- Think carefully about what counts as a datapoint in your project before designing accessors.

## Grouping and Iteration

- Ungrouped dataset length = row count.
- Grouped dataset length = unique group count.
- Grouping changes what "one datapoint" means for iteration and splitting.
- `group_labels` follow current grouping; `index_as_tuples()` always reflects raw rows.

## Subsetting and Splits

- `get_subset(...)` accepts exactly one selector mode at a time.
- Use `groupby(...)` when train/test splitting should happen on a higher level than raw rows.
- Use `create_string_group_labels(...)` for `GroupKFold` and similar sklearn splitters.
- If the dataset is already grouped, `create_string_group_labels(...)` columns must be a subset of `groupby_cols`.

## Common Mistakes

- Non-deterministic index creation.
- Loading full signals/dataframes in `create_index()`.
- Forgetting `groupby_cols`/`subset_index` in custom dataset init.
- Accessing per-recording data from a subset that still contains multiple rows/groups.
- Splitting raw rows when the real independence unit is participant/session/day.

## Minimal Pattern

```python
class MyDataset(Dataset[MyGroupLabel]):
    def __init__(self, root: Path, groupby_cols=None, subset_index=None):
        self.root = root
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self) -> pd.DataFrame:
        return build_index(self.root).sort_values(["participant", "recording"]).reset_index(drop=True)

    @property
    def signal(self) -> pd.DataFrame:
        self.assert_is_single(None, "signal")
        return load_signal(...)
```

## Source of Truth

- `https://tpcp.readthedocs.io/en/latest/auto_examples/datasets/_01_datasets_basics.html`
- `https://tpcp.readthedocs.io/en/latest/guides/algorithms_pipelines_datasets.html`
- API docs for `Dataset`
