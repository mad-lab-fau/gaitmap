"""Test the dataset helpers."""
from typing import List, Callable
import numpy as np
import pandas as pd

import pytest

from gaitmap.utils.consts import SF_COLS, SF_GYR, SF_ACC, BF_COLS, BF_GYR, BF_ACC
from gaitmap.utils.dataset_helper import (
    _has_sf_cols,
    _has_bf_cols,
    is_single_sensor_dataset,
    is_multi_sensor_dataset,
    get_multi_sensor_dataset_names,
)


def _create_test_multiindex():
    return pd.MultiIndex.from_product([list("abc"), list("123")])


@pytest.fixture(params=(("both", True, True), ("acc", True, False), ("gyr", False, True)))
def combinations(request):
    return request.param


@pytest.fixture(params=("any", "body", "sensor"))
def frame(request):
    return request.param


class TestColumnHelper:

    method: Callable
    cols: List[str]
    gyr_cols: List[str]
    acc_cols: List[str]

    @pytest.fixture(params=["sf", "bf"], autouse=True)
    def select_method(self, request):
        self.method = {"sf": _has_sf_cols, "bf": _has_bf_cols}[request.param]
        self.cols = {"sf": SF_COLS, "bf": BF_COLS}[request.param]
        self.gyr_cols = {"sf": SF_GYR, "bf": BF_GYR}[request.param]
        self.acc_cols = {"sf": SF_ACC, "bf": BF_ACC}[request.param]

    def test_columns_correct(self):
        assert self.method(self.cols)

    def test_gyr_columns_only(self):
        assert self.method(self.gyr_cols, check_acc=False)

    def test_acc_columns_only(self):
        assert self.method(self.acc_cols, check_gyr=False)

    def test_missing_columns(self):
        assert not self.method(self.acc_cols)

    def test_wrong_names(self):
        assert not self.method(list(range(6)))

    def test_missing_acc_columns(self):
        assert not self.method(self.acc_cols[:-1], check_gyr=False)

    def test_missing_gyr_columns(self):
        assert not self.method(self.gyr_cols[:-1], check_acc=False)


class TestIsSingleSensorDataset:
    @pytest.mark.parametrize(
        "value",
        ({"test": pd.DataFrame}, list(range(6)), "test", np.arange(6), pd.DataFrame(columns=_create_test_multiindex())),
    )
    def test_wrong_datatype(self, value):
        assert not is_single_sensor_dataset(value, check_acc=False, check_gyr=False)

    def test_correct_datatype(self):
        assert is_single_sensor_dataset(pd.DataFrame(), check_acc=False, check_gyr=False)

    @pytest.mark.parametrize(
        "cols, frame_valid, col_check_valid",
        (
            (SF_COLS, "sensor", "both"),
            (BF_COLS, "body", "both"),
            (BF_GYR, "body", "gyr"),
            (BF_ACC, "body", "acc"),
            (SF_GYR, "sensor", "gyr"),
            (SF_ACC, "sensor", "acc"),
        ),
    )
    def test_correct_columns(self, cols, frame_valid, col_check_valid, combinations, frame):
        """Test all possible combinations of inputs."""
        col_check, check_acc, check_gyro = combinations
        output = is_single_sensor_dataset(
            pd.DataFrame(columns=cols), check_acc=check_acc, check_gyr=check_gyro, frame=frame
        )

        valid_frame = (frame_valid == frame) or (frame == "any")
        valid_cols = (col_check == col_check_valid) or (col_check_valid == "both")
        expected_outcome = valid_cols and valid_frame

        assert output == expected_outcome

    def test_invalid_frame_argument(self):
        with pytest.raises(ValueError):
            is_single_sensor_dataset(pd.DataFrame(), frame="invalid_value")


class TestIsMultiSensorDataset:
    @pytest.mark.parametrize(
        "value", (list(range(6)), "test", np.arange(6), {}, pd.DataFrame(), pd.DataFrame(columns=[*range(3)])),
    )
    def test_wrong_datatype(self, value):
        assert not is_multi_sensor_dataset(value, check_acc=False, check_gyr=False)

    def test_correct_datatype(self):
        assert is_multi_sensor_dataset(
            pd.DataFrame([[*range(9)]], columns=_create_test_multiindex()), check_acc=False, check_gyr=False
        )

    @pytest.mark.parametrize(
        "cols, frame_valid, col_check_valid",
        (
            (SF_COLS, "sensor", "both"),
            (BF_COLS, "body", "both"),
            (BF_GYR, "body", "gyr"),
            (BF_ACC, "body", "acc"),
            (SF_GYR, "sensor", "gyr"),
            (SF_ACC, "sensor", "acc"),
        ),
    )
    def test_correct_columns(self, cols, frame_valid, col_check_valid, combinations, frame):
        """Test all possible combinations of inputs."""
        col_check, check_acc, check_gyro = combinations
        output = is_multi_sensor_dataset(
            pd.DataFrame([[*range(len(cols) * 2)]], columns=pd.MultiIndex.from_product((("a", "b"), cols))),
            check_acc=check_acc,
            check_gyr=check_gyro,
            frame=frame,
        )

        valid_frame = (frame_valid == frame) or (frame == "any")
        valid_cols = (col_check == col_check_valid) or (col_check_valid == "both")
        expected_outcome = valid_cols and valid_frame

        assert output == expected_outcome

    def test_invalid_frame_argument(self):
        with pytest.raises(ValueError):
            is_multi_sensor_dataset(
                pd.DataFrame([[*range(9)]], columns=_create_test_multiindex()), frame="invalid_value"
            )


class TestGetMultiSensorDatasetNames:
    @pytest.mark.parametrize("obj", ({"a": [], "b": [], "c": []}, pd.DataFrame(columns=_create_test_multiindex())))
    def test_names_simple(self, obj):
        assert set(get_multi_sensor_dataset_names(obj)) == {"a", "b", "c"}
