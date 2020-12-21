from operator import itemgetter
from typing import Union, Optional

import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, index: Optional[Union[pd.Index, pd.MultiIndex]] = None, select_lvl: int = 0):
        self.index = index
        self.select_lvl = select_lvl

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value: Optional[Union[pd.Index, pd.MultiIndex]]):
        if value is None:
            self._index = self._create_index()
        else:
            if value.names.count(None) == len(value.names):
                value.set_names(tuple(["index_lvl_{}".format(i) for i in range(len(value[0]))]), inplace=True)

            self._index = value

    @property
    def select_lvl(self):
        return self._select_lvl

    @select_lvl.setter
    def select_lvl(self, value: int):
        self._select_lvl = value
        self.columns = None

    @property
    def columns(self):
        if self.select_lvl == 0:
            return self._columns.values

        return list(set(map(itemgetter(self.select_lvl), self._columns.values)))

    @columns.setter
    def columns(self, _: None):
        self._columns = pd.Series(
            self.index.to_frame().reset_index(drop=True).groupby(self.index.names[: self.select_lvl + 1]).groups.keys()
        )

    @property
    def shape(self):
        return [len(self._columns)]

    def __getitem__(self, subscript):
        where_to_search = self.index.tolist()
        filtered_list = []

        if isinstance(subscript, (tuple, np.ndarray)):
            if isinstance(subscript[0], str):
                filtered_list = list(filter(lambda x: x[: self.select_lvl + 1][-1] in subscript, where_to_search))
            else:
                what_to_search = (
                    [tuple([x]) for x in self._columns.iloc[list(subscript)].to_list()]
                    if self.select_lvl == 0
                    else self._columns.iloc[list(subscript)].to_list()
                )

                filtered_list = list(filter(lambda x: x[: self.select_lvl + 1] in what_to_search, where_to_search))

        elif isinstance(subscript, str):
            filtered_list = (
                list(filter(lambda x: x[0] == subscript, where_to_search))
                if self.select_lvl == 0
                else list(filter(lambda x: x[: self.select_lvl + 1][-1] == subscript, where_to_search))
            )

        if len(filtered_list) == 0:
            raise IndexError("Subscript {} not applicable to this dataset!".format(subscript))

        return Dataset(self._make_multi_index(filtered_list), select_lvl=self.select_lvl,)

    def __repr__(self):
        return str(self.index)

    def to_multi_index(self):
        return self.index

    def to_frame(self):
        return self.index.to_frame().reset_index(drop=True)

    def _make_multi_index(self, value: list):
        return pd.MultiIndex.from_arrays(
            [list(map(itemgetter(i), value)) for i in range(len(value[0]))], names=self.index.names
        )

    @staticmethod
    def _create_index():
        raise ValueError("A dataset requires a pd.MultiIndex that it should represent!")
