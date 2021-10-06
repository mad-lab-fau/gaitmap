"""Base class for all datasets."""
from typing import Iterator, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

from gaitmap.base import _BaseSerializable

Self = TypeVar("Self", bound="Dataset")


class Dataset(_BaseSerializable):
    """Baseclass for gaitmap Dataset objects.

    This class provides fundamental functionality like iteration, getting subsets and compatibility with sklearn's
    cross validation helper.

    For more information check out the examples and user guides on datasets.

    Parameters
    ----------
    groupby_cols
        A column name or a list of column names that should be used to group the index before iterating over it.
        For examples see below.
    subset_index
        For all classes that inherit from this class, subset_index **must** be None.
        The subset_index **must** be created in the method __create_index.
        If the base class is used, then the index the dataset should represent **must** be a pd.Dataframe
        containig the index.
        For examples see below.

    Attributes
    ----------
    index
        The index of the dataset.
        This returns either the `subset_index` or the base index returned by `create_index`.
    grouped_index
        The index, but all groupby columns are represented as MultiIndex.
        Note, that the order can be different as the order of index.
    groups
        Returns all possible combinations based on the specified `groupby` columns.
        If `groupby` is None, this returns the row indices.
        These are also the groups/indices used when iterating over the dataset.
        The groups are sorted by name.
    shape
        Represents the number of all groups encapsulated in a tuple.
        This is only necessary if `sklearn.model_selection.KFold` is used for splitting the dataset.

    Examples
    --------
    This class is usually not meant to be used directly, but the following code snippets show some common operations
    that can be expected to work for all dataset subclasses.

    >>> import pandas as pd
    >>> from itertools import product
    >>>
    >>> test_index = pd.DataFrame(
    ...     list(product(("patient_1", "patient_2", "patient_3"), ("test_1", "test_2"), ("1", "2"))),
    ...     columns=["patient", "test", "extra"],
    ... )
    >>> # We create a little dummy dataset by passing an index directly to `test_index`
    >>> dataset = Dataset(subset_index=test_index)
    >>> dataset
    Dataset [12 groups/rows]
    <BLANKLINE>
             patient    test extra
       0   patient_1  test_1     1
       1   patient_1  test_1     2
       2   patient_1  test_2     1
       3   patient_1  test_2     2
       4   patient_2  test_1     1
       5   patient_2  test_1     2
       6   patient_2  test_2     1
       7   patient_2  test_2     2
       8   patient_3  test_1     1
       9   patient_3  test_1     2
       10  patient_3  test_2     1
       11  patient_3  test_2     2

    We can loop over the dataset.
    By default we will loop over each row.

    >>> for r in dataset[:2]:
    ...     print(r)
    Dataset [1 groups/rows]
    <BLANKLINE>
            patient    test extra
       0  patient_1  test_1     1
    Dataset [1 groups/rows]
    <BLANKLINE>
            patient    test extra
       0  patient_1  test_1     2

    We can also change `groupby` (either in the init or afterwards), to loop over other combinations.
    If we select the level `test`, we will loop over all `patient`-`test` combinations.

    >>> grouped_dataset = dataset.groupby(["patient", "test"])
    >>> grouped_dataset  # doctest: +NORMALIZE_WHITESPACE
    Dataset [6 groups/rows]
    <BLANKLINE>
                           patient    test extra
       patient   test
       patient_1 test_1  patient_1  test_1     1
                 test_1  patient_1  test_1     2
                 test_2  patient_1  test_2     1
                 test_2  patient_1  test_2     2
       patient_2 test_1  patient_2  test_1     1
                 test_1  patient_2  test_1     2
                 test_2  patient_2  test_2     1
                 test_2  patient_2  test_2     2
       patient_3 test_1  patient_3  test_1     1
                 test_1  patient_3  test_1     2
                 test_2  patient_3  test_2     1
                 test_2  patient_3  test_2     2

    >>> for r in grouped_dataset[:2]:
    ...     print(r)  # doctest: +NORMALIZE_WHITESPACE
    Dataset [1 groups/rows]
    <BLANKLINE>
                           patient    test extra
       patient   test
       patient_1 test_1  patient_1  test_1     1
                 test_1  patient_1  test_1     2
    Dataset [1 groups/rows]
    <BLANKLINE>
                           patient    test extra
       patient   test
       patient_1 test_2  patient_1  test_2     1
                 test_2  patient_1  test_2     2

    To iterate over the unique values of a specific the "iter_level" function:

    >>> for r in list(grouped_dataset.iter_level("patient"))[:2]:
    ...     print(r)  # doctest: +NORMALIZE_WHITESPACE
    Dataset [2 groups/rows]
    <BLANKLINE>
                           patient    test extra
       patient   test
       patient_1 test_1  patient_1  test_1     1
                 test_1  patient_1  test_1     2
                 test_2  patient_1  test_2     1
                 test_2  patient_1  test_2     2
    Dataset [2 groups/rows]
    <BLANKLINE>
                           patient    test extra
       patient   test
       patient_2 test_1  patient_2  test_1     1
                 test_1  patient_2  test_1     2
                 test_2  patient_2  test_2     1
                 test_2  patient_2  test_2     2

    We can also get arbitary subsets from the dataset:

    >>> subset = grouped_dataset.get_subset(patient=["patient_1", "patient_2"], extra="2")
    >>> subset  # doctest: +NORMALIZE_WHITESPACE
    Dataset [4 groups/rows]
    <BLANKLINE>
                           patient    test extra
       patient   test
       patient_1 test_1  patient_1  test_1     2
                 test_2  patient_1  test_2     2
       patient_2 test_1  patient_2  test_1     2
                 test_2  patient_2  test_2     2

    If we want to use datasets in combination with `sklearn.model_selection.GroupKFold`, we can generate valid group
    labels as follows.
    Note, that you usually don't want to use that in combination with `self.groupby`.

    >>> # We are using the ungrouped dataset again!
    >>> group_labels = dataset.create_group_labels(["patient", "test"])
    >>> pd.concat([dataset.index, pd.Series(group_labels, name="group_labels")], axis=1)
          patient    test extra         group_labels
    0   patient_1  test_1     1  (patient_1, test_1)
    1   patient_1  test_1     2  (patient_1, test_1)
    2   patient_1  test_2     1  (patient_1, test_2)
    3   patient_1  test_2     2  (patient_1, test_2)
    4   patient_2  test_1     1  (patient_2, test_1)
    5   patient_2  test_1     2  (patient_2, test_1)
    6   patient_2  test_2     1  (patient_2, test_2)
    7   patient_2  test_2     2  (patient_2, test_2)
    8   patient_3  test_1     1  (patient_3, test_1)
    9   patient_3  test_1     2  (patient_3, test_1)
    10  patient_3  test_2     1  (patient_3, test_2)
    11  patient_3  test_2     2  (patient_3, test_2)

    """

    groupby_cols: Optional[Union[List[str], str]]
    subset_index: Optional[pd.DataFrame]

    def __init__(
        self,
        *,
        groupby_cols: Optional[Union[List[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ):
        self.groupby_cols = groupby_cols
        self.subset_index = subset_index

    @property
    def index(self) -> pd.DataFrame:
        """Get index."""
        if self.subset_index is None:
            return self.create_index()

        return self.subset_index

    @property
    def groups(self) -> List[Union[str, Tuple[str, ...]]]:
        """Get all groups based on the set groupby level."""
        return self._get_unique_groups().to_list()

    def __len__(self) -> int:
        """Get the length of the dataset.

        This is equal to the number of rows in the index, if `self.groupby_cols=None`.
        Otherwise, it is equal to the number of unique groups.
        """
        return len(self.groups)

    @property
    def shape(self) -> Tuple[int]:
        """Get the shape of the dataset.

        This only reports a single dimension.
        This is equal to the number of rows in the index, if `self.groupby_cols=None`.
        Otherwise, it is equal to the number of unique groups.
        """
        return (len(self),)

    @property
    def grouped_index(self) -> pd.DataFrame:
        """Return the the index with the `groupby` columns set as multiindex."""
        if self.groupby_cols is None:
            groupby_cols = self.index.columns.to_list()
        else:
            groupby_cols = self.groupby_cols
        try:
            return self.index.set_index(groupby_cols, drop=False)
        except KeyError as e:
            raise KeyError(
                "You can only groupby columns that are part of the index columns ({}) and not {}".format(
                    list(self.index.columns), self.groupby_cols
                )
            ) from e

    def _get_unique_groups(self) -> Union[pd.MultiIndex, pd.Index]:
        return self.grouped_index.index.unique()

    def __getitem__(self: Self, subscript: Union[int, Sequence[int], np.ndarray]) -> Self:
        """Return a dataset object containing only the selected row indices of `self.groups`."""
        multi_index = self._get_unique_groups()[subscript]
        if not isinstance(multi_index, pd.Index):
            multi_index = [multi_index]

        return self.clone().set_params(subset_index=self.grouped_index.loc[multi_index].reset_index(drop=True))

    def groupby(self, groupby_cols: Optional[Union[List[str], str]]):
        """Return a copy of the dataset grouped by the specified columns.

        Each unique group represents a single data point in the resulting dataset.

        Parameters
        ----------
        groupby_cols
            None (no grouping) or a valid subset of the columns available in the dataset index.

        """
        grouped_ds = self.clone().set_params(groupby_cols=groupby_cols)
        # Get grouped index to raise an error here, in case the `groupby_cols` are invalid.
        _ = grouped_ds.grouped_index
        return grouped_ds

    def get_subset(
        self: Self,
        *,
        index: Optional[pd.DataFrame] = None,
        bool_map: Optional[Sequence[bool]] = None,
        **kwargs: Optional[Union[List[str], str]],
    ) -> Self:
        """Get a subset of the dataset.

        Parameters
        ----------
        index
            `pd.DataFrame` that is a valid subset of the current dataset index.
        bool_map
            bool-map that is used to index the current index-dataframe.
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
                    (index, bool_map, kwargs),
                )
            ).count(False)
            > 1
        ):
            raise ValueError("Only one of `selected_keys`, `index`, `bool_map` or kwarg can be set!")

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
            _assert_all_in_df(self.index, kwargs)

            subset_index = self.index.loc[self.index[list(kwargs.keys())].isin(kwargs).all(axis=1)].reset_index(
                drop=True
            )
            if len(subset_index) == 0:
                raise KeyError(f"No datapoint in the dataset matched the following filter: {kwargs}")

            return self.clone().set_params(subset_index=subset_index)

        raise ValueError("At least one of `selected_keys`, `index`, `bool_map` or kwarg must not be None!")

    def __repr__(self) -> str:
        """Return string representation of the dataset object."""
        if self.groupby_cols is None:
            repr_index = self.index
        else:
            repr_index = self.grouped_index
        return "{} [{} groups/rows]\n\n   {}\n\n   ".format(
            self.__class__.__name__,
            self.shape[0],
            str(repr_index).replace("\n", "\n   "),
        )[:-5]

    def _repr_html_(self) -> str:
        """Return html representation of the dataset object."""
        if self.groupby_cols is None:
            repr_index = self.index
        else:
            repr_index = self.grouped_index
        return '<h4 style="margin-bottom: 0.1em;">{} [{} groups/rows]</h3>\n'.format(
            self.__class__.__name__, self.shape[0]
        ) + repr_index._repr_html_().replace("<div>", '<div style="margin-top: 0em">').replace(
            '<table border="1" class="dataframe"', '<table style="margin-left: 3em;"'
        ).replace(
            "<th>", '<th style="text-align: center;">'
        ).replace(
            "<td>", '<td style="text-align: center; padding-left: 2em; padding-right: 2em;">'
        )

    def __iter__(self: Self) -> Iterator[Self]:
        """Return generator object containing a subset for every combination up to and including the selected level."""
        return (self.__getitem__(i) for i in range(self.shape[0]))

    def iter_level(self: Self, level: str) -> Iterator[Self]:
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
        if level not in self.index.columns:
            raise ValueError(f"`level` must be one of {list(self.index.columns)}")

        return (self.get_subset(**{level: category}) for category in self.index[level].unique())

    def is_single(self, groupby_cols: Optional[Union[str, List[str]]]) -> bool:
        """Return True if index contains only one row/group with the given groupby settings.

        If `groupby_cols=None` this checks if there is only a single row left.

        Parameters
        ----------
        groupby_cols
            None (no grouping) or a valid subset of the columns available in the dataset index.

        """
        return self.groupby(groupby_cols).shape[0] == 1

    def assert_is_single(self, groupby_cols: Optional[Union[str, List[str]]], property_name) -> None:
        """Raise error if index does contain more than one group/row with the given groupby settings.

        This should be used when implementing access to data values, which can only be accessed when only a single
        trail/participant/etc. exist in the dataset.

        Parameters
        ----------
        groupby_cols
            None (no grouping) or a valid subset of the columns available in the dataset index.
        property_name
            Name of the property this check is used in.
            Used to format the error message.

        """
        if not self.is_single(groupby_cols):
            if groupby_cols is None:
                groupby_cols = self.index.columns.to_list()
            raise ValueError(
                "The data value {} of dataset {} can only be accessed if there is only a single "
                "combination of the columns {} left in a data-subset".format(
                    property_name, self.__class__.__name__, groupby_cols
                )
            )

    def create_group_labels(self, label_cols: Union[str, List[str]]):
        """Generate a list of labels for each group/row in the dataset.

        Note that this has a different usecase than the dataset-wide groupby.
        Using `groupby` reduces the effective size of the dataset to the number of groups.
        This method produces a group label for each group/row that is already in the dataset, without changing the
        dataset.

        The output of this method can be used in combination with `sklearn.model_selection.GroupKFold` as the group
        label.

        Parameters
        ----------
        label_cols
            The columns that should be included in the label.
            If the dataset is already grouped, this must be a subset of  `self.groupby_cols`.

        """
        unique_index = self._get_unique_groups().to_frame()
        try:
            return unique_index.set_index(label_cols).index.to_list()
        except KeyError as e:
            if self.groupby_cols is not None:
                raise KeyError(
                    "When using `create_group_labels` with a grouped dataset, the selected columns must "
                    "be a subset of `self.groupby_cols` ({}) and not ({})".format(self.groupby_cols, label_cols)
                ) from e
            raise KeyError(
                "The selected label columns ({}) are not in the index of the dataset ({})".format(
                    label_cols, list(self.index.columns)
                )
            ) from e

    def create_index(self) -> pd.DataFrame:
        """Create the full index for the dataset.

        This needs to be implemented by the subclass.

        """
        raise NotImplementedError()


def _ensure_is_list(x):
    return x if isinstance(x, list) else [x]


def _assert_all_in_df(df, dic):
    """Check that all values of the dictionary are in the column 'key' of the pandas dataframe."""
    for key, value in dic.items():
        try:
            index_level = df[key]
        except KeyError as e:
            raise KeyError(f"Can not filter by key `{key}`! Key must be one of {list(df.columns)}!") from e
        if not set(value).issubset(index_level):
            raise KeyError(f"At least one of {value} is not in level {key}")
