"""Some custom helper types to make typehints and typechecking easier.

For user facing type declarations, please see `gaitmap.utils.datatype_helper`.
"""
from typing import TYPE_CHECKING, Any, Hashable, TypeVar, Union

import pandas as pd

_Hashable = Union[Hashable, str]
_HashableVar = TypeVar("_HashableVar", Hashable, str)

_DataFrame = Any if TYPE_CHECKING else pd.DataFrame
