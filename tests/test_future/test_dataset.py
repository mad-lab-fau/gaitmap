from itertools import product
from operator import itemgetter

import pandas as pd
import numpy as np
import pytest

from gaitmap.future.dataset import Dataset


def _create_valid_index(input_dict=None, columns_names=None):
    if input_dict is None:
        test_multi_index = pd.DataFrame(
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

        test_multi_index["patients"] = test_multi_index["patients"].astype(
            pd.CategoricalDtype(["patient_1", "patient_2", "patient_3"])
        )

        test_multi_index["tests"] = test_multi_index["tests"].astype(
            pd.CategoricalDtype(["test_1", "test_2", "test_3"])
        )

        test_multi_index["extra"] = test_multi_index["extra"].astype(pd.CategoricalDtype(["0", "1"]))

        return test_multi_index

    output = {column_name: [] for column_name in columns_names}

    for key, value in input_dict.items():
        combinations = list(product(*([[key]] + list(map(itemgetter(1), value.items())))))

        for i in range(len(combinations[0])):
            for val in map(itemgetter(i), combinations):
                output[columns_names[i]].append(val)

    output = pd.DataFrame(output)

    for column_name in columns_names:
        output[column_name] = output[column_name].astype(pd.CategoricalDtype(output[column_name].unique()))

    return output


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
        "selected_keys,index,bool_map,what_to_expect",
        [
            (
                ["patient_1"],
                None,
                None,
                _create_valid_index(
                    {"patient_1": {"a": ["test_1", "test_2"], "b": ["0", "1"]}},
                    columns_names=["patients", "tests", "extra"],
                ),
            ),
            (
                None,
                _create_valid_index(
                    {"patient_1": {"a": ["test_1", "test_2"], "b": ["0", "1"]}},
                    columns_names=["patients", "tests", "extra"],
                ),
                None,
                _create_valid_index(
                    {"patient_1": {"a": ["test_1", "test_2"], "b": ["0", "1"]}},
                    columns_names=["patients", "tests", "extra"],
                ),
            ),
            (
                None,
                None,
                _create_random_bool_map(12, 68752868),
                _create_valid_index()[_create_random_bool_map(12, 68752868)].reset_index(drop=True),
            ),
        ],
    )
    def test_get_subset(self, selected_keys, index, bool_map, what_to_expect):
        pd.testing.assert_frame_equal(
            what_to_expect,
            Dataset(subset_index=_create_valid_index()).get_subset(selected_keys, index, bool_map).index_as_dataframe(),
            check_categorical=False,
        )
