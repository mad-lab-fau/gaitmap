"""Backend-neutral helpers for HMM repr/clone support."""

from __future__ import annotations

from typing import Any

try:
    import pomegranate as pg
except ImportError:  # pragma: no cover - exercised in environments without pomegranate
    pg = None

from tpcp import BaseTpcpObject, CloneFactory


def is_serialized_hmm_state(value: Any) -> bool:
    return (
        hasattr(value, "compiled")
        and hasattr(value, "trained_with")
        and callable(getattr(value, "to_json", None))
        and callable(getattr(type(value), "from_json", None))
    )


class ShortenedHMMPrint(BaseTpcpObject):
    """Mixin class to better format HMM models when printing them."""

    def __repr_parameter__(self, name: str, value: Any) -> str:
        if name == "model":
            if is_serialized_hmm_state(value):
                n_states = len(value.compiled.state_names) if getattr(value, "compiled", None) is not None else "?"
                backend = getattr(getattr(value, "trained_with", None), "backend_id", "?")
                return f"{name}=HMMState[backend={backend}, states={n_states}](...)"
            if pg is not None and isinstance(value, pg.HiddenMarkovModel):
                return f"{name}=HiddenMarkovModel[name={value.name}](...)"
            if (
                pg is not None
                and isinstance(value, CloneFactory)
                and isinstance(value.default_value, pg.HiddenMarkovModel)
            ):
                return f"{name}=cf(HiddenMarkovModel[name={value.get_value().name}](...))"
        return super().__repr_parameter__(name, value)
