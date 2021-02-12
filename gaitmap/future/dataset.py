"""Base class for all datasets."""
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
        containig the index.
        For examples see below.
    select_lvl
        The level that should be used for indexing the dataset.
        This **must** be a string corresponding to one of the columns of the index.
        If left empty the last column is set as indexing level.
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

    >>> dataset.get_subset(tests=["test_1", "test_3"]) # doctest: +NORMALIZE_WHITESPACE
    Dataset
        index [8 rows x 3 columns] =
    <BLANKLINE>
                    patients   tests extra
                0  patient_1  test_1     0
                1  patient_1  test_1     1
                2  patient_2  test_1     0
                3  patient_2  test_1     1
                4  patient_3  test_1     0
                5  patient_3  test_1     1
                6  patient_3  test_3     0
                7  patient_3  test_3     1
    <BLANKLINE>

    >>> dataset.select_lvl = "tests"
    >>> next(dataset.__iter__())  # doctest: +NORMALIZE_WHITESPACE
    Dataset
        index [2 rows x 3 columns] =
    <BLANKLINE>
                    patients   tests extra
                0  patient_1  test_1     0
                1  patient_1  test_1     1
    <BLANKLINE>

    >>> next(dataset.iter_level()) # doctest: +NORMALIZE_WHITESPACE
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

    >>> next(dataset.iter_level(level="extra")) # doctest: +NORMALIZE_WHITESPACE
    Dataset
        index [6 rows x 3 columns] =
    <BLANKLINE>
                    patients   tests extra
                0  patient_1  test_1     0
                1  patient_1  test_2     0
                2  patient_2  test_1     0
                3  patient_3  test_1     0
                4  patient_3  test_2     0
                5  patient_3  test_3     0
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
    def column_combinations(self) -> Union[pd.MultiIndex, pd.Index]:
        """Get all possible combinations up to and including the selected level."""
        return self._index_helper.index.unique()

    @property
    def shape(self) -> Tuple[int]:
        """Get shape."""
        return (len(self.column_combinations),)

    @property
    def _index_helper(self) -> pd.DataFrame:
        columns = list(self.index.columns)
        return self.index.set_index(columns[: columns.index(self._get_selected_level()) + 1], drop=False)

    def _get_selected_level(self) -> str:
        if self.select_lvl is None:
            return self.index.columns[-1]

        if self.select_lvl in self.index.columns:
            return self.select_lvl

        raise ValueError("`select_lvl` must be one of {}".format(self.index.columns.to_list()))

    def __getitem__(self: Self, subscript: Union[int, List[int]]) -> Self:
        """Return a dataset object by passing subscript to loc."""
        multi_index = self.column_combinations[subscript]

        return self.clone().set_params(
            subset_index=self._index_helper.loc[
                [multi_index] if isinstance(multi_index, (tuple, str)) else multi_index
            ].reset_index(drop=True)
        )

    def get_subset(
        self: Self,
        selected_keys: Optional[Union[List[str], str]] = None,
        index: Optional[pd.DataFrame] = None,
        bool_map: Optional[List[bool]] = None,
        **kwargs: Optional[List[str]],
    ) -> Self:
        """Get a subset of the dataset.

        Parameters
        ----------
        selected_keys
            String or list of strings corresponding to the categories of the selected level that should be selected.
        index
            `pd.DataFrame` that is a valid subset of the current dataset index.
        bool_map
            List of booleans that will be passed to the index for filtering.
            The list **must** be of same length as the number of rows in the index.
        **kwargs
            The key **must** be the name of a index column.
            The value is a list containing strings that correspond to the categories that should be kept.
            For examples see above.

        Returns
        -------
        subset
            New dataset object filtered by specified parameter.

        """
        if (
            list(
                map(
                    lambda x: x is None or (isinstance(x, dict) and len(x) == 0),
                    (selected_keys, index, bool_map, kwargs),
                )
            ).count(False)
            > 1
        ):
            raise ValueError("Only one of `selected_keys`, `index`, `bool_map` or kwarg can be set!")

        if selected_keys is not None:
            # We just pretend we set kwargs with the correct name
            kwargs = {self._get_selected_level(): selected_keys}

        if index is not None:
            if len(index) == 0:
                raise ValueError("Provided index is not formatted correctly. Make sure it is not empty!")

            return self.clone().set_params(subset_index=index.reset_index(drop=True))

        if bool_map is not None:
            if len(bool_map) != self.index.shape[0]:
                raise ValueError(f"Parameter bool_map must have length {self.index.shape[0]} but has {len(bool_map)}!")

            return self.clone().set_params(subset_index=self.index[bool_map].reset_index(drop=True))

        if len(kwargs) > 0:
            kwargs = {k: _ensure_is_list(v) for k, v in kwargs.items()}

            # Check if all values are actually in their respective columns.
            # This is not strictly required, but avoids user error
            assert_all_in_df(self.index, kwargs)

            subset_index = self.index.loc[self.index[list(kwargs.keys())].isin(kwargs).all(axis=1)].reset_index(
                drop=True
            )
            if len(subset_index) == 0:
                raise KeyError(f"No datapoint in the dataset matched the following filter: {kwargs}")

            return self.clone().set_params(subset_index=subset_index)

        raise ValueError("At least one of `selected_keys`, `index`, `bool_map` or kwarg must not be None!")

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
            '<h3 style="margin-bottom: -0.5em;">{}</h3>\n'.format(self.__class__.__name__)
            + '<h4 style="margin-left: 2.5em;">index [{} rows x {} columns] =</h4>'.format(
                self.index.shape[0], self.index.shape[1]
            )
            + self.index._repr_html_()
            .replace("<div>", '<div style="margin-top: 0em">')
            .replace('<table border="1" class="dataframe"', '<table style="font-size: 16px; margin-left: 3em;"')
            .replace("<th>", '<th style="text-align: center;">')
            .replace("<td>", '<td style="text-align: center; padding-left: 2em; padding-right: 2em;">')
        )

    def __iter__(self: Self) -> Generator[Self, None, None]:
        """Return generator object containing a subset for every combination up to and including the selected level."""
        return (self.__getitem__(i) for i in range(self.shape[0]))

    def iter_level(self: Self, level: Optional[str] = None) -> Generator[Self, None, None]:
        """Return generator object containing a subset for every category from the selected level.

        Parameters
        ----------
        level
            Optional `str` that sets the level which shall be used for iterating.
            This **must** be one of the columns names of the index.

        Returns
        -------
        subset
            New dataset object containing only one category in the specified `level`.

        """
        if level and level not in self.index.columns:
            raise ValueError(f"`level` must be one of {list(self.index.columns)}")

        level = level or self._get_selected_level()
        return (self.get_subset(**{level: category}) for category in self.index[level].unique())

    def is_single(self) -> bool:
        """Return True if index contains only one row else False."""
        return self.shape[0] == 1

    def _create_index(self) -> pd.DataFrame:
        raise NotImplementedError()


def _ensure_is_list(x):
    return x if isinstance(x, list) else [x]


def assert_all_in_df(df, dic):
    """Check that all values of the dictionary are in the column 'key' of the pandas dataframe."""
    for key, value in dic.items():
        try:
            index_level = df[key]
        except KeyError as e:
            raise KeyError(f"Can not filter by key `{key}`! Key must be one of {list(df.columns)}!") from e
        if not set(value).issubset(index_level):
            raise KeyError(f"At least one of {value} is not in level {key}")
