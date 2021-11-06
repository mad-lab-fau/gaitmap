"""A set of helper functions to make developing algorithms easier."""
from typing import Any, Dict, TypeVar

from gaitmap.base import _BaseSerializable
from gaitmap.utils._types import _Hashable, _HashableVar

Algo = TypeVar("Algo", bound=_BaseSerializable)


def invert_result_dictionary(
    nested_dict: Dict[_Hashable, Dict[_HashableVar, Any]]
) -> Dict[_HashableVar, Dict[_Hashable, Any]]:
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
    out: Dict[_HashableVar, Dict[_Hashable, Any]] = {}
    for ok, ov in nested_dict.items():
        for k, v in ov.items():
            nested = out.setdefault(k, {})
            nested[ok] = v
    return out


def set_params_from_dict(obj: Any, param_dict: Dict[str, Any], result_formatting: bool = False):
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
        if result_formatting is True:
            if not k.endswith("_"):
                k += "_"
        setattr(obj, k, v)


def default(algo: Algo) -> Algo:
    algo.__DEFAULT = True
    return algo
