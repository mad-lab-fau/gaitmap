"""A set of helper functions to make developing algorithms easier."""
from typing import Hashable, Dict, Any


def invert_result_dictionary(nested_dict: Dict[Hashable, Dict[str, Any]]) -> Dict[Hashable, Dict[Hashable, Any]]:
    """Invert result dictionaries that are obtained from multi sensor results."""
    out = dict()
    for ok, ov in nested_dict.items():
        for k, v in ov.items():
            nested = out.setdefault(k, dict())
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
