"""Base class for all datasets."""

from typing import Optional

import numpy as np
import pandas as pd

from gaitmap.base import _BaseSerializable


class Dataset(_BaseSerializable):
    """This is the base class for all datasets.

    Attributes
    ----------
    subset_index : Optional[pd.Dataframe]
        For all classes that inherit from this class, subset_index **must** be None.
        The subset_index **must** be created in the method __create_index.
        If the base class is used, then the index the dataset should represent **must** be a pd.Dataframe
        containig the index. Every column of said pd.Dataframe **must** be of type pd.CategoricalDtype
        to represent every possible state of that column.
        For examples see below.
    select_lvl : Optional[str]
        The level of the index which should be used for indexing. This must be a string corresponding
        to one of the columns of the index.
        If left empty the first column is set as indexing level.

    Parameters
    ----------
    index
        The index of the dataset. Internally it is stored as a pd.Dataframe.
    select_lvl
        The select_lvl property sets the desired level which shall be indexed.
    columns
        A dict where the keys are the categories for the selected level and the values are
        lists of corresponding indices. For examples see below.
    shape
        Represents the length of the indexed level encapsulated in a list. This is only
        necessary if sklearn.model_selection.KFold is used for splitting the dataset.

    Examples
    --------
    >>> test_index = pd.DataFrame({"patients": ["patient_1","patient_1","patient_1","patient_1","patient_2","patient_2","patient_3","patient_3","patient_3","patient_3","patient_3", "patient_3",],"tests": ["test_1","test_1","test_2","test_2","test_1","test_1","test_1","test_1","test_2","test_2","test_3","test_3",],"extra": ["0", "1", "0", "1", "0", "1", "0", "1", "0", "1", "0", "1"]}) # noqa: E501
    >>> test_index["patients"] = test_index["patients"].astype(pd.CategoricalDtype(["patient_1", "patient_2", "patient_3"])) # noqa: E501
    >>> test_index["tests"] = test_index["tests"].astype(pd.CategoricalDtype(["test_1", "test_2", "test_3"]))
    >>> test_index["extra"] = test_index["extra"].astype(pd.CategoricalDtype(["0", "1"]))
    >>> test_index
         patients   tests extra
    0   patient_1  test_1     0
    1   patient_1  test_1     1
    2   patient_1  test_2     0
    3   patient_1  test_2     1
    4   patient_2  test_1     0
    5   patient_2  test_1     1
    6   patient_3  test_1     0
    7   patient_3  test_1     1
    8   patient_3  test_2     0
    9   patient_3  test_2     1
    10  patient_3  test_3     0
    11  patient_3  test_3     1

    >>> dataset = Dataset(test_index, "tests")
    >>> dataset.columns
    {'test_1': [0, 1, 4, 5, 6, 7], 'test_2': [2, 3, 8, 9], 'test_3': [10, 11]}

    >>> dataset.select_lvl = "patients"
    >>> dataset["patient_2"]
        patients   tests extra
    0  patient_2  test_1     0
    1  patient_2  test_1     1

    >>> dataset["patient_1"].index_as_multi_index()
    MultiIndex([('patient_1', 'test_1', '0'),
                ('patient_1', 'test_1', '1'),
                ('patient_1', 'test_2', '0'),
                ('patient_1', 'test_2', '1')],
               names=['patients', 'tests', 'extra'])

    """

    def __init__(self, subset_index: Optional[pd.DataFrame] = None, select_lvl: Optional[str] = None):
        self.__subset_index = subset_index
        self.select_lvl = list(self.index.columns)[0] if select_lvl is None else select_lvl

    @property
    def index(self):
        """Get index."""
        if hasattr(self, "__cached_index"):
            return self.__cached_index

        return self.__create_index() if self.__subset_index is None else self.__subset_index

    @property
    def select_lvl(self):  # noqa: missing-function-docstring
        return self._select_lvl

    @select_lvl.setter
    def select_lvl(self, value: str):
        """Set select_lvl."""
        if value in self.index.columns:
            self._select_lvl = value
        else:
            raise ValueError("select_lvl must be one of {}".format(self.index.columns.to_list()))

    @property
    def columns(self):
        """Get columns."""
        return self.index.groupby(by=self.select_lvl).groups

    @property
    def shape(self):
        """Get shape."""
        return [len(np.concatenate(list(self.columns.values())))]

    def __getitem__(self, subscript):
        """Return a dataset object."""
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

        return Dataset(pd.concat(to_concat).reset_index(drop=True), self.select_lvl)

    def __repr__(self):
        """Return string representation of the dataset object."""
        return str(self.index)

    def index_as_multi_index(self):
        """Return the dataset as a pd.MultiIndex."""
        return pd.MultiIndex.from_frame(self.index)

    def index_as_dataframe(self):
        """Return the dataset as a pd.Dataframe."""
        return self.index

    def __is_single(self):
        not_zero = 0

        for value in self.columns.values():
            not_zero += 1 if len(value) > 0 else 0

        return not_zero <= 1

    def __iter__(self):
        """Return same object and initialize private attributes for iteration."""
        self.__n = 0  # noqa: attribute-defined-outside-init
        self.__categories_to_search = self.index[self.select_lvl].cat.categories  # noqa: attribute-defined-outside-init
        return self

    def __next__(self):
        """Return new dataset objects by splitting the initial object by the categories of the selected level."""
        if self.__is_single() or self.__n == len(self.__categories_to_search):
            raise StopIteration

        to_return = self.__getitem__(self.__categories_to_search[self.__n])
        self.__n += 1
        return to_return

    @staticmethod
    def __create_index():
        raise NotImplementedError
