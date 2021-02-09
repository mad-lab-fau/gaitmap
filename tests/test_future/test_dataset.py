from itertools import product
from operator import itemgetter

import pandas as pd
import numpy as np
import pytest

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
                ["patient_1"],
                None,
                None,
                None,
                _create_valid_index(
                    {"patient_1": {"a": ["test_1", "test_2"], "b": ["0", "1"]}},
                    columns_names=["patients", "tests", "extra"],
                ),
                None,
            ),
            (
                None,
                _create_valid_index(
                    {"patient_1": {"a": ["test_1", "test_2"], "b": ["0", "1"]}},
                    columns_names=["patients", "tests", "extra"],
                ),
                None,
                None,
                _create_valid_index(
                    {"patient_1": {"a": ["test_1", "test_2"], "b": ["0", "1"]}},
                    columns_names=["patients", "tests", "extra"],
                ),
                None,
            ),
            (
                None,
                None,
                _create_random_bool_map(12, 68752868),
                None,
                _create_valid_index()[_create_random_bool_map(12, 68752868)].reset_index(drop=True),
                None,
            ),
            (
                None,
                None,
                None,
                {"patients": ["patient_1", "patient_3"], "tests": ["test_2", "test_3"], "extra": ["0"]},
                _create_valid_index(
                    {
                        "patient_1": {"a": ["test_2"], "b": ["0"]},
                        "patient_3": {"a": ["test_2", "test_3"], "b": ["0"]},
                    },
                    columns_names=["patients", "tests", "extra"],
                ),
                None,
            ),
            (
                None,
                pd.DataFrame(),
                None,
                None,
                "Provided index is not formatted correctly",
                ValueError,
            ),
            (
                None,
                None,
                _create_random_bool_map(12, 68752868)[:-1],
                None,
                "Parameter bool_map must have length",
                ValueError,
            ),
            (
                None,
                None,
                None,
                None,
                "At least one of selected_keys, index, bool_map or kwarg must be not None!",
                ValueError,
            ),
            (
                ["wrong", "patient_1"],
                None,
                None,
                None,
                "Can not filter by",
                KeyError,
            ),
            (
                "wrong",
                None,
                None,
                None,
                "Can not filter by",
                KeyError,
            ),
        ],
    )
    def test_get_subset(self, selected_keys, index, bool_map, kwargs, what_to_expect, expect_error):
        df = Dataset(subset_index=_create_valid_index())

        if expect_error is not None:
            with pytest.raises(expect_error, match=what_to_expect):
                df.get_subset(selected_keys, index, bool_map) if kwargs is None else df.get_subset(**kwargs)
        else:
            pd.testing.assert_frame_equal(
                left=what_to_expect,
                right=df.get_subset(selected_keys, index, bool_map).index_as_dataframe()
                if kwargs is None
                else df.get_subset(**kwargs).index_as_dataframe(),
                check_categorical=False,
            )

    @pytest.mark.parametrize(
        "select_lvl,what_to_expect,expect_error",
        [(None, "patients", False), ("tests", "tests", False), ("xyz", "select_lvl must be one of", True)],
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
