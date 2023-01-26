"""A set of helper functions to make developing algorithms easier."""
from __future__ import annotations

from typing import Any

from gaitmap.utils._types import _Hashable, _HashableVar


def invert_result_dictionary(
    nested_dict: dict[_Hashable, dict[_HashableVar, Any]]
) -> dict[_HashableVar, dict[_Hashable, Any]]:
    """Invert result dictionaries that are obtained from multi sensor results.

    This method expects a two level dictionary and flips the levels.
    This means that if a value can be accessed as `nested_dict[k1][k2] = v` in the input, it can be accessed as
    `output_dict[k2][k1] = v`.

    Examples
    --------
    >>> in_dict = {"level_1_1": {"level_2_1": "val_2_1", "level_2_2": "val_2_2"}, "level_1_2": {"level_2_3": "val_2_3"}}
    >>> from pprint import pprint
    >>> pprint(invert_result_dictionary(in_dict))
    {'level_2_1': {'level_1_1': 'val_2_1'},
     'level_2_2': {'level_1_1': 'val_2_2'},
     'level_2_3': {'level_1_2': 'val_2_3'}}

    """
    out: dict[_HashableVar, dict[_Hashable, Any]] = {}
    for ok, ov in nested_dict.items():
        for k, v in ov.items():
            nested = out.setdefault(k, {})
            nested[ok] = v
    return out


def set_params_from_dict(obj: Any, param_dict: dict[str, Any], result_formatting: bool = False):
    """Update object attributes from dictionary.

    The object will be updated inplace.

    Parameters
    ----------
    obj
        The gaitmap obj to update
    param_dict
        The dictionary of new values to set/update
    result_formatting
        If True all keys will get a trailing "_", if they don't have one already.
        This marks them as "results" based on the gaitmap guidelines.

    """
    for k, v in param_dict.items():
        if result_formatting is True and not k.endswith("_"):
            k += "_"
        setattr(obj, k, v)
