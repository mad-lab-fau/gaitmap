"""Some custom helper types to make typehints and typechecking easier.

For user facing type declarations, please see `gaitmap.utils.datatype_helper`.
"""
from typing import Union, TYPE_CHECKING, Any, Hashable, TypeVar

import pandas as pd

_Hashable = Union[Hashable, str]
_HashableVar = TypeVar("_HashableVar", Hashable, str)

if TYPE_CHECKING:
    _DataFrame = Any
else:
    _DataFrame = pd.DataFrame
