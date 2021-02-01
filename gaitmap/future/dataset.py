"""Base class for all datasets."""
from functools import reduce
from operator import and_
from typing import Optional, List, Union, Sequence, Generator, Tuple, TypeVar

import pandas as pd

from gaitmap.base import _BaseSerializable

Self = TypeVar("Self", bound="Dataset")


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
        The level that should be interpreted as categories which will be used for indexing the dataset.
        This must be a string corresponding to one of the columns of the index.
        If left empty the first column is set as indexing level.
        For examples see below.

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
    >>> dataset.columns
    {'patient_1': [0, 1, 2, 3], 'patient_2': [4, 5], 'patient_3': [6, 7, 8, 9, 10, 11]}

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

    def __init__(
        self,
        subset_index: Optional[pd.DataFrame] = None,
        select_lvl: Optional[str] = None,
        level_order: Optional[List[str]] = None,
    ):
        self.level_order = level_order
        self.subset_index = subset_index
        self.select_lvl = select_lvl

    @property
    def index(self) -> pd.DataFrame:
        """Get index."""
        if self.subset_index is None:
            return self._create_index()

        return self.subset_index if self.level_order is None else self.subset_index[self.level_order]

    @property
    def column_combinations(self) -> Union[List[str], List[Tuple[str]]]:
        """Get all possible combinations up to the selected level."""
        columns = list(self.index.columns)
        return [key for key, _ in self.index.groupby(columns[: columns.index(self._get_selected_level()) + 1])]

    @property
    def shape(self) -> Tuple[int]:
        """Get shape."""
        return (len(self.column_combinations),)

    def _get_selected_level(self):
        if self.select_lvl is None:
            return self.index.columns[0]

        if self.select_lvl in self.index.columns:
            return self.select_lvl

        raise ValueError("select_lvl must be one of {}".format(self.index.columns.to_list()))

    def __getitem__(self: Self, subscript) -> Self:
        """Return a dataset object."""
        return self.clone().set_params(subset_index=self.index.iloc[subscript])

    def get_subset(
        self,
        selected_keys: Optional[Union[Sequence[str], str]] = None,
        index: Optional[pd.DataFrame] = None,
        bool_map: Optional[List[bool]] = None,
        **kwargs: Optional[List[str]],
    ) -> Self:
        """Return a dataset object."""
        if selected_keys is not None:
            return self.clone().set_params(
                subset_index=self.index.loc[
                    self.index[self._get_selected_level()].isin(_ensure_is_list(selected_keys))
                ].reset_index(drop=True)
            )

        if index is not None:
            if all(map(lambda x: isinstance(x, pd.CategoricalDtype), index.dtypes)) and len(index) > 0:
                return self.clone().set_params(subset_index=index.reset_index(drop=True))

            raise ValueError(
                "Provided index is not formatted correctly. Make sure it is not empty and that all columns are of "
                "dtype pd.CategoricalDtype!"
            )

        if bool_map is not None:
            if len(bool_map) != self.shape[0]:
                raise ValueError(f"Parameter bool_map must have length {self.shape[0]} but has {len(bool_map)}!")

            return self.clone().set_params(subset_index=self.index[bool_map].reset_index(drop=True))

        if len(kwargs) > 0:
            return self.clone().set_params(
                subset_index=self.index.loc[
                    reduce(and_, (self.index[key].isin(_ensure_is_list(value)) for key, value in kwargs.items()))
                ].reset_index(drop=True)
            )

        raise ValueError("At least one of selected_keys, index, bool_map or kwarg must be not None!")

    def __repr__(self) -> str:
        """Return string representation of the dataset object."""
        return "{}\n\tindex [{} rows x {} columns] =\n\n\t\t{}\n\n\t".format(
            self.__class__.__name__,
            self.index.shape[0],
            self.index.shape[1],
            str(self.index).replace("\n", "\n\t\t"),
        )[:-2]

    def _repr_html_(self) -> str:
        """Return html representation of the dataset object."""
        return (
            "<h3 style='margin-bottom: -1.5em'>{}</h3>".format(self.__class__.__name__)
            + "<h4 style='margin-left: 2.5em'>index [{} rows x {} columns] =</h4>".format(
                self.index.shape[0], self.index.shape[1]
            )
            + self.index._repr_html_()
            .replace("<div>", "<div style='margin-top: 0em'>")
            .replace("<table", "<table style='width:100%;'")
            .replace("text-align: right;", "text-align: middle;")
        )

    def index_as_multi_index(self) -> pd.MultiIndex:
        """Return the dataset as a pd.MultiIndex."""
        return pd.MultiIndex.from_frame(self.index)

    def index_as_dataframe(self) -> pd.DataFrame:
        """Return the dataset as a pd.Dataframe."""
        return self.index

    def __iter__(self: Self) -> Generator[Self, None, None]:
        """Return generator object containing subset of every combination up to and including the selected level."""
        columns = list(self.index.columns)

        return (
            self.clone().set_params(subset_index=group)
            for _, group in self.index.groupby(columns[: columns.index(self._get_selected_level()) + 1])
        )

    def iter(self: Self) -> Generator[Self, None, None]:
        """Return generator object containing subset of every category from the selected level."""
        return (self.get_subset(category) for category in self.index.groupby(self._get_selected_level()).groups)

    def _create_index(self) -> pd.DataFrame:
        raise NotImplementedError


def _ensure_is_list(x):
    return x if isinstance(x, list) else [x]
