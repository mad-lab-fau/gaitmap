from typing import Optional

import numpy as np
import pandas as pd

from gaitmap.base import _BaseSerializable


class Dataset(_BaseSerializable):
    def __init__(self, select_lvl: str, subset_index: Optional[pd.DataFrame] = None):
        self.__subset_index = subset_index
        self.select_lvl = select_lvl

    @property
    def index(self):
        if self.__subset_index is None:
            return self._create_index()
        return self.__subset_index

    @property
    def select_lvl(self):
        return self._select_lvl

    @select_lvl.setter
    def select_lvl(self, value: str):
        if value in self.index.columns:
            self._select_lvl = value
        else:
            raise ValueError("select_lvl must be one of {}".format(self.index.columns.to_list()))

    @property
    def columns(self):
        return self.index.groupby(by=self.select_lvl).groups

    @property
    def shape(self):
        return [len(np.concatenate(list(self.columns.values())))]

    def __getitem__(self, subscript):
        to_concat = []

        if isinstance(subscript, (tuple, list, np.ndarray)):
            if isinstance(subscript[0], str):
                for string in subscript:
                    to_concat.append(self.index.loc[self.columns[string]])
            else:
                to_concat.append(
                    self.index.iloc[list(filter(lambda x: x in np.concatenate(list(self.columns.values())), subscript))]
                )

        elif isinstance(subscript, str):
            to_concat.append(self.index.loc[self.columns[subscript]])

        if len(to_concat) == 0:
            raise IndexError("Subscript {} not applicable to this dataset!".format(subscript))

        return Dataset(self.select_lvl, pd.concat(to_concat).reset_index(drop=True))

    def __repr__(self):
        return str(self.index)

    def index_as_multi_index(self):
        return pd.MultiIndex.from_frame(self.index)

    def index_as_dataframe(self):
        return self.index

    @staticmethod
    def _create_index():
        raise NotImplementedError
