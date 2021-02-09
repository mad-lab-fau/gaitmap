"""Base class for all datasets."""
from functools import reduce
from operator import and_
from typing import Optional, List, Union, Generator, Tuple, TypeVar

import pandas as pd

from gaitmap.base import _BaseSerializable

Self = TypeVar("Self", bound="Dataset")


class Dataset(_BaseSerializable):
    """This is the base class for all datasets.

    Parameters
    ----------
    subset_index
        For all classes that inherit from this class, subset_index **must** be None.
        The subset_index **must** be created in the method __create_index.
        If the base class is used, then the index the dataset should represent **must** be a pd.Dataframe
        containig the index. Every column of said pd.Dataframe **must** be of type pd.CategoricalDtype
        to represent every possible state of that column.
        For examples see below.
    select_lvl
        The level that should be used for indexing the dataset.
        This **must** be a string corresponding to one of the columns of the index.
        If left empty the first column is set as indexing level.
        For examples see below.
    level_order
        List containing **all** columns of the index in an arbitrary order.
        If this is not Null it will be used for reordering the index.

    Attributes
    ----------
    index
        The index of the dataset. Internally it is stored as a pd.Dataframe.
    select_lvl
        The select_lvl property sets the desired level up to which it should index the dataset.
    column_combinations
        Returns all possible combinations up to the selected level.
    shape
        Represents the number of all column_combinations encapsulated in a tuple. This is only
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

    >>> dataset = Dataset(test_index, "extra")
    >>> dataset.column_combinations
    MultiIndex([('patient_1', 'test_1', '0'),
                ('patient_1', 'test_1', '1'),
                ('patient_1', 'test_2', '0'),
                ('patient_1', 'test_2', '1'),
                ('patient_2', 'test_1', '0'),
                ('patient_2', 'test_1', '1'),
                ('patient_3', 'test_1', '0'),
                ('patient_3', 'test_1', '1'),
                ('patient_3', 'test_2', '0'),
                ('patient_3', 'test_2', '1'),
                ('patient_3', 'test_3', '0'),
                ('patient_3', 'test_3', '1')],
               names=['patients', 'tests', 'extra'])

    >>> dataset.select_lvl = "patients"
    >>> list(dataset.column_combinations)
    ['patient_1', 'patient_2', 'patient_3']

    >>> dataset.get_subset(selected_keys="patient_2") # doctest: +NORMALIZE_WHITESPACE
    Dataset
        index [2 rows x 3 columns] =
    <BLANKLINE>
                patients   tests extra
            0  patient_2  test_1     0
            1  patient_2  test_1     1
    <BLANKLINE>

    >>> dataset.get_subset(tests=["test_1", "test_3"]).index_as_multi_index()
    MultiIndex([('patient_1', 'test_1', '0'),
                ('patient_1', 'test_1', '1'),
                ('patient_2', 'test_1', '0'),
                ('patient_2', 'test_1', '1'),
                ('patient_3', 'test_1', '0'),
                ('patient_3', 'test_1', '1'),
                ('patient_3', 'test_3', '0'),
                ('patient_3', 'test_3', '1')],
               names=['patients', 'tests', 'extra'])

    >>> dataset.select_lvl = "tests"
    >>> next(dataset.__iter__())  # doctest: +NORMALIZE_WHITESPACE
    Dataset
        index [2 rows x 3 columns] =
    <BLANKLINE>
                    patients   tests extra
                0  patient_1  test_1     0
                1  patient_1  test_1     1
    <BLANKLINE>

    >>> next(dataset.iter()) # doctest: +NORMALIZE_WHITESPACE
    Dataset
        index [6 rows x 3 columns] =
    <BLANKLINE>
                    patients   tests extra
                0  patient_1  test_1     0
                1  patient_1  test_1     1
                2  patient_2  test_1     0
                3  patient_2  test_1     1
                4  patient_3  test_1     0
                5  patient_3  test_1     1
    <BLANKLINE>

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
    def column_combinations(self) -> Union[pd.MultiIndex, pd.CategoricalIndex]:
        """Get all possible combinations up to and including the selected level."""
        return self._index_helper.index.unique()

    @property
    def shape(self) -> Tuple[int]:
        """Get shape."""
        return (len(self.column_combinations),)

    @property
    def _index_helper(self):
        columns = list(self.index.columns)
        return self.index.set_index(columns[: columns.index(self._get_selected_level()) + 1], drop=False)

    def _get_selected_level(self):
        if self.select_lvl is None:
            return self.index.columns[0]

        if self.select_lvl in self.index.columns:
            return self.select_lvl

        raise ValueError("select_lvl must be one of {}".format(self.index.columns.to_list()))

    def __getitem__(self: Self, subscript) -> Self:
        """Return a dataset object by passing subscript to loc."""
        return self.clone().set_params(
            subset_index=self._index_helper.loc[self.column_combinations[subscript]].reset_index(drop=True)
        )

    def get_subset(
        self: Self,
        selected_keys: Optional[Union[List[str], str]] = None,
        index: Optional[pd.DataFrame] = None,
        bool_map: Optional[List[bool]] = None,
        **kwargs: Optional[List[str]],
    ) -> Self:
        """Return a dataset object.

        Parameters
        ----------
        selected_keys
            String or list of strings corresponding to the categories of the selected level
            that should **not** be filtered out.
        index
            If index is not None it will be checked for validity and will be set
            as the index of the returned dataset object.
        bool_map
            List of booleans that will be passed to the index for filtering. The list **must**
            be of same length as the number of rows in the index.
        **kwargs
            The key **must** be the name of a index column.
            The value is a list containing strings that correspond to the categories that should be kept.
            For examples see above.

        Returns
        -------
        subset
            New dataset object filtered by specified parameter.

        """
        if selected_keys is not None:
            selected_keys = _ensure_is_list(selected_keys)
            index_at_selected_lvl = self.index[self._get_selected_level()]

            not_in_index_uniques = [key for key in selected_keys if key not in index_at_selected_lvl.unique()]
            if len(not_in_index_uniques) > 0:
                raise KeyError(
                    "Can not filter by {}! The keys used to filter must be one of {}!".format(
                        not_in_index_uniques, list(index_at_selected_lvl.unique())
                    )
                )

            return self.clone().set_params(
                subset_index=self.index.loc[index_at_selected_lvl.isin(selected_keys)].reset_index(drop=True)
            )

        if index is not None:
            if all(map(lambda x: isinstance(x, pd.CategoricalDtype), index.dtypes)) and len(index) > 0:
                return self.clone().set_params(subset_index=index.reset_index(drop=True))

            raise ValueError(
                "Provided index is not formatted correctly. Make sure it is not empty and that all columns are of "
                "dtype pd.CategoricalDtype!"
            )

        if bool_map is not None:
            if len(bool_map) != self.index.shape[0]:
                raise ValueError(f"Parameter bool_map must have length {self.index.shape[0]} but has {len(bool_map)}!")

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
        """Return generator object containing a subset for every combination up to and including the selected level."""
        return (self.__getitem__(i) for i in range(self.shape[0]))

    def iter(self: Self) -> Generator[Self, None, None]:
        """Return generator object containing a subset for every category from the selected level."""
        return (self.get_subset(selected_keys=category) for category in self.index[self._get_selected_level()].unique())

    def _is_single(self):
        return self.shape[0] == 1

    def _create_index(self) -> pd.DataFrame:
        raise NotImplementedError


def _ensure_is_list(x):
    return x if isinstance(x, list) else [x]
