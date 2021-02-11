from itertools import product
from operator import itemgetter

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from gaitmap.future.dataset import Dataset


def _create_valid_index(input_dict=None, columns_names=None):
    if input_dict is None:
        return pd.DataFrame(
            {
                "patients": [
                    "patient_1",
                    "patient_1",
                    "patient_1",
                    "patient_1",
                    "patient_2",
                    "patient_2",
                    "patient_3",
                    "patient_3",
                    "patient_3",
                    "patient_3",
                    "patient_3",
                    "patient_3",
                ],
                "tests": [
                    "test_1",
                    "test_1",
                    "test_2",
                    "test_2",
                    "test_1",
                    "test_1",
                    "test_1",
                    "test_1",
                    "test_2",
                    "test_2",
                    "test_3",
                    "test_3",
                ],
                "extra": ["0", "1", "0", "1", "0", "1", "0", "1", "0", "1", "0", "1"],
            }
        )

    output = {column_name: [] for column_name in columns_names}

    for key, value in input_dict.items():
        combinations = list(product(*([[key]] + list(map(itemgetter(1), value.items())))))

        for i in range(len(combinations[0])):
            for val in map(itemgetter(i), combinations):
                output[columns_names[i]].append(val)

    return pd.DataFrame(output)


def _create_random_bool_map(n, seed):
    np.random.seed(seed)
    return list(map(lambda x: x >= 0.5, np.random.rand(n)))


class TestDataset:
    @pytest.mark.parametrize(
        "select_lvl,what_to_expect,lvl_order",
        [
            ("patients", 3, None),
            ("tests", 6, None),
            ("extra", 12, None),
            ("patients", 12, ["extra", "tests", "patients"]),
            ("tests", 3, ["tests", "extra", "patients"]),
            ("extra", 12, ["tests", "patients", "extra"]),
        ],
    )
    def test_indexing_with_optional_lvl_order(self, select_lvl, what_to_expect, lvl_order):
        df = (
            Dataset(subset_index=_create_valid_index(), select_lvl=select_lvl)
            if lvl_order is None
            else Dataset(subset_index=_create_valid_index(), select_lvl=select_lvl, level_order=lvl_order)
        )

        assert df.shape[0] == what_to_expect

    @pytest.mark.parametrize(
        "selected_keys,index,bool_map,kwargs,what_to_expect,expect_error",
        [
            (
                None,
                None,
                None,
                None,
                "At least one of `selected_keys`, `index`, `bool_map` or kwarg must not be None!",
                ValueError,
            ),
            (
                None,
                _create_valid_index(),
                _create_random_bool_map(12, 432),
                None,
                "Only one of `selected_keys`, `index`, `bool_map` or kwarg can be set!",
                ValueError,
            ),
        ],
    )
    def test_get_subset_generic_errors(self, selected_keys, index, bool_map, kwargs, what_to_expect, expect_error):
        df = Dataset(subset_index=_create_valid_index())

        with pytest.raises(expect_error, match=what_to_expect):
            df.get_subset(selected_keys, index, bool_map) if kwargs is None else df.get_subset(
                selected_keys, index, bool_map, **kwargs
            )

    @pytest.mark.parametrize(
        "selected_keys,what_to_expect,expect_error",
        [
            (
                ["0"],
                _create_valid_index(
                    {
                        "patient_1": {"a": ["test_1", "test_2"], "b": ["0"]},
                        "patient_2": {"a": ["test_1"], "b": ["0"]},
                        "patient_3": {"a": ["test_1", "test_2", "test_3"], "b": ["0"]},
                    },
                    columns_names=["patients", "tests", "extra"],
                ),
                False,
            ),
            (
                "wrong",
                "Can not filter by",
                True,
            ),
            (
                ["wrong", "1"],
                "Can not filter by",
                True,
            ),
        ],
    )
    def test_get_subset_selected_keys(self, selected_keys, what_to_expect, expect_error):
        df = Dataset(subset_index=_create_valid_index())

        if expect_error:
            with pytest.raises(KeyError, match=what_to_expect):
                df.get_subset(selected_keys)
        else:
            pd.testing.assert_frame_equal(left=what_to_expect, right=df.get_subset(selected_keys=selected_keys).index)

    @pytest.mark.parametrize(
        "index,what_to_expect,expect_error",
        [
            (
                _create_valid_index(
                    {"patient_1": {"a": ["test_1", "test_2"], "b": ["0", "1"]}},
                    columns_names=["patients", "tests", "extra"],
                ),
                _create_valid_index(
                    {"patient_1": {"a": ["test_1", "test_2"], "b": ["0", "1"]}},
                    columns_names=["patients", "tests", "extra"],
                ),
                False,
            ),
            (
                pd.DataFrame(),
                "Provided index is not formatted correctly",
                True,
            ),
        ],
    )
    def test_get_subset_index(self, index, what_to_expect, expect_error):
        df = Dataset(subset_index=_create_valid_index())

        if expect_error:
            with pytest.raises(ValueError, match=what_to_expect):
                df.get_subset(index=index)
        else:
            pd.testing.assert_frame_equal(left=what_to_expect, right=df.get_subset(index=index).index)

    @pytest.mark.parametrize(
        "bool_map,what_to_expect,expect_error",
        [
            (
                _create_random_bool_map(12, 68752868),
                _create_valid_index()[_create_random_bool_map(12, 68752868)].reset_index(drop=True),
                False,
            ),
            (
                _create_random_bool_map(12, 68752868)[:-1],
                "Parameter bool_map must have length",
                True,
            ),
        ],
    )
    def test_get_subset_bool_map(self, bool_map, what_to_expect, expect_error):
        df = Dataset(subset_index=_create_valid_index())

        if expect_error:
            with pytest.raises(ValueError, match=what_to_expect):
                df.get_subset(bool_map=bool_map)
        else:
            pd.testing.assert_frame_equal(left=what_to_expect, right=df.get_subset(bool_map=bool_map).index)

    @pytest.mark.parametrize(
        "kwargs,what_to_expect,expect_error",
        [
            (
                {"patients": ["patient_1", "patient_3"], "tests": ["test_2", "test_3"], "extra": ["0"]},
                _create_valid_index(
                    {
                        "patient_1": {"a": ["test_2"], "b": ["0"]},
                        "patient_3": {"a": ["test_2", "test_3"], "b": ["0"]},
                    },
                    columns_names=["patients", "tests", "extra"],
                ),
                False,
            ),
            (
                {"wrong": ["patient_1", "patient_3"], "tests": ["test_2", "test_3"], "extra": ["0"]},
                "Can not filter by",
                True,
            ),
        ],
    )
    def test_get_subset_kwargs(self, kwargs, what_to_expect, expect_error):
        df = Dataset(subset_index=_create_valid_index())

        if expect_error:
            with pytest.raises(KeyError, match=what_to_expect):
                df.get_subset(**kwargs)
        else:
            pd.testing.assert_frame_equal(left=what_to_expect, right=df.get_subset(**kwargs).index)

    @pytest.mark.parametrize(
        "subscript,select_lvl,what_to_expect,expect_error",
        [
            (
                0,
                "extra",
                _create_valid_index(
                    {
                        "patient_1": {"a": ["test_1"], "b": ["0"]},
                    },
                    columns_names=["patients", "tests", "extra"],
                ),
                False,
            ),
            (
                4,
                "patients",
                "out of bounds",
                True,
            ),
            (
                [0, 1, 4],
                "patients",
                "out of bounds",
                True,
            ),
            (
                [0, 1],
                "patients",
                _create_valid_index(
                    {
                        "patient_1": {"a": ["test_1", "test_2"], "b": ["0", "1"]},
                        "patient_2": {"a": ["test_1"], "b": ["0", "1"]},
                    },
                    columns_names=["patients", "tests", "extra"],
                ),
                False,
            ),
            (
                [0, 4],
                "tests",
                _create_valid_index(
                    {
                        "patient_1": {"a": ["test_1"], "b": ["0", "1"]},
                        "patient_3": {"a": ["test_2"], "b": ["0", "1"]},
                    },
                    columns_names=["patients", "tests", "extra"],
                ),
                False,
            ),
        ],
    )
    def test_getitem(self, subscript, select_lvl, what_to_expect, expect_error):
        df = Dataset(subset_index=_create_valid_index(), select_lvl=select_lvl)

        if expect_error:
            with pytest.raises(IndexError, match=what_to_expect):
                _ = df[subscript]
        else:
            pd.testing.assert_frame_equal(left=what_to_expect, right=df[subscript].index)

    @pytest.mark.parametrize(
        "select_lvl,what_to_expect,expect_error",
        [(None, "extra", False), ("tests", "tests", False), ("xyz", "`select_lvl` must be one of", True)],
    )
    def test_get_selected_lvl(self, select_lvl, what_to_expect, expect_error):
        df = (
            Dataset(_create_valid_index(), select_lvl=select_lvl)
            if select_lvl is not None
            else Dataset(_create_valid_index())
        )

        if expect_error:
            with pytest.raises(ValueError, match=what_to_expect):
                df._get_selected_level()
        else:
            assert df._get_selected_level() == what_to_expect

    @pytest.mark.parametrize(
        "index,what_to_expect",
        [
            (
                _create_valid_index(
                    {
                        "patient_1": {"a": ["test_1"], "b": ["0", "1"]},
                        "patient_3": {"a": ["test_2"], "b": ["0", "1"]},
                    },
                    columns_names=["patients", "tests", "extra"],
                ),
                False,
            ),
            (
                _create_valid_index(
                    {"patient_1": {"a": ["test_1"], "b": ["0"]}},
                    columns_names=["patients", "tests", "extra"],
                ),
                True,
            ),
        ],
    )
    def test_is_single(self, index, what_to_expect):
        assert Dataset(subset_index=index).is_single() == what_to_expect

    def test__create_index_call(self):
        with pytest.raises(NotImplementedError):
            _ = Dataset().index

    @pytest.mark.parametrize(
        "n_splits,select_lvl,what_to_expect,expect_error",
        [
            (13, "extra", "Cannot have number of splits", True),
            (
                5,
                "tests",
                (
                    _create_valid_index(
                        {
                            "patient_2": {"a": ["test_1"], "b": ["0", "1"]},
                            "patient_3": {"a": ["test_1", "test_2", "test_3"], "b": ["0", "1"]},
                        },
                        columns_names=["patients", "tests", "extra"],
                    ),
                    _create_valid_index(
                        {
                            "patient_1": {"a": ["test_1", "test_2"], "b": ["0", "1"]},
                        },
                        columns_names=["patients", "tests", "extra"],
                    ),
                ),
                False,
            ),
        ],
    )
    def test_dataset_with_kfold(self, n_splits, select_lvl, what_to_expect, expect_error):
        df = Dataset(_create_valid_index(), select_lvl=select_lvl)

        if expect_error:
            with pytest.raises(ValueError, match=what_to_expect):
                next(KFold(n_splits=n_splits).split(df))
        else:
            train, test = next(KFold(n_splits=n_splits).split(df))
            pd.testing.assert_frame_equal(left=what_to_expect[0], right=df[train].index)
            pd.testing.assert_frame_equal(left=what_to_expect[1], right=df[test].index)

    @pytest.mark.parametrize(
        "select_lvl,what_to_expect",
        [
            (
                "patients",
                _create_valid_index(
                    {
                        "patient_1": {"a": ["test_1", "test_2"], "b": ["0", "1"]},
                    },
                    columns_names=["patients", "tests", "extra"],
                ),
            ),
            (
                "tests",
                _create_valid_index(
                    {
                        "patient_1": {"a": ["test_1"], "b": ["0", "1"]},
                    },
                    columns_names=["patients", "tests", "extra"],
                ),
            ),
            (
                "extra",
                _create_valid_index(
                    {
                        "patient_1": {"a": ["test_1"], "b": ["0"]},
                    },
                    columns_names=["patients", "tests", "extra"],
                ),
            ),
        ],
    )
    def test_iter(self, select_lvl, what_to_expect):
        df = Dataset(_create_valid_index(), select_lvl=select_lvl)
        pd.testing.assert_frame_equal(left=what_to_expect, right=next(df.__iter__()).index)

    @pytest.mark.parametrize(
        "level,what_to_expect,expect_error",
        [
            (
                None,
                _create_valid_index(
                    {
                        "patient_1": {"a": ["test_1", "test_2"], "b": ["0"]},
                        "patient_2": {"a": ["test_1"], "b": ["0"]},
                        "patient_3": {"a": ["test_1", "test_2", "test_3"], "b": ["0"]},
                    },
                    columns_names=["patients", "tests", "extra"],
                ),
                False,
            ),
            (
                "extra",
                _create_valid_index(
                    {
                        "patient_1": {"a": ["test_1", "test_2"], "b": ["0"]},
                        "patient_2": {"a": ["test_1"], "b": ["0"]},
                        "patient_3": {"a": ["test_1", "test_2", "test_3"], "b": ["0"]},
                    },
                    columns_names=["patients", "tests", "extra"],
                ),
                False,
            ),
            (
                "tests",
                _create_valid_index(
                    {
                        "patient_1": {"a": ["test_1"], "b": ["0", "1"]},
                        "patient_2": {"a": ["test_1"], "b": ["0", "1"]},
                        "patient_3": {"a": ["test_1"], "b": ["0", "1"]},
                    },
                    columns_names=["patients", "tests", "extra"],
                ),
                False,
            ),
            (
                "patients",
                _create_valid_index(
                    {
                        "patient_1": {"a": ["test_1", "test_2"], "b": ["0", "1"]},
                    },
                    columns_names=["patients", "tests", "extra"],
                ),
                False,
            ),
            (
                "wrong",
                "`level` must be one of",
                True,
            ),
        ],
    )
    def test_iter_level(self, level, what_to_expect, expect_error):
        df = Dataset(_create_valid_index())

        if expect_error:
            with pytest.raises(ValueError, match=what_to_expect):
                next(df.iter_level(level=level))
        else:
            # Get the original level value that should be restored
            original_level = df._get_selected_level()
            # Create the generator
            generator = df.iter_level(level=level)
            # See if the first subset is what we expect to get
            pd.testing.assert_frame_equal(left=what_to_expect, right=next(generator).index)
            # See if the level is the same as level whilst the generator is "not done" yet
            assert df._get_selected_level() == level if level is not None else df.index.columns[-1]
            # "Force" the generator to run till the end
            _ = list(generator)
            # See if the level has been correctly restored
            assert df._get_selected_level() == original_level
