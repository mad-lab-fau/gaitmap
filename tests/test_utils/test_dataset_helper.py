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
    is_single_sensor_stride_list,
    is_multi_sensor_stride_list,
)


def _create_test_multiindex():
    return pd.MultiIndex.from_product([list("abc"), list("123")])


@pytest.fixture(params=(("both", True, True), ("acc", True, False), ("gyr", False, True)))
def combinations(request):
    return request.param


@pytest.fixture(params=("any", "body", "sensor"))
def frame(request):
    return request.param


@pytest.fixture(params=("any", "min_vel"))
def stride_types(request):
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


class TestIsSingleSensorStrideList:
    @pytest.mark.parametrize(
        "value",
        (
            list(range(6)),
            "test",
            np.arange(6),
            {},
            pd.DataFrame(),
            pd.DataFrame(columns=[*range(3)]),
            pd.DataFrame([[*range(9)]], columns=_create_test_multiindex()),
        ),
    )
    def test_wrong_datatype(self, value):
        assert not is_single_sensor_stride_list(value)

    @pytest.mark.parametrize(
        "cols, stride_types_valid",
        (
            (["s_id", "start", "end", "gsd_id"], "any"),
            (["s_id", "start", "end", "gsd_id", "something_extra"], "any"),
            (["s_id", "start", "end", "gsd_id", "pre_ic", "ic", "min_vel", "tc"], "min_vel"),
            (["s_id", "start", "end", "gsd_id", "pre_ic", "ic", "min_vel", "tc", "something_extra"], "min_vel"),
        ),
    )
    def test_valid_versions(self, cols, stride_types_valid, stride_types):
        expected_outcome = stride_types == stride_types_valid or stride_types == "any"

        out = is_single_sensor_stride_list(pd.DataFrame(columns=cols), stride_type=stride_types)

        assert expected_outcome == out

    @pytest.mark.parametrize(
        "start, min_vel, expected",
        ((np.arange(10), np.arange(10), True), (np.arange(10), np.arange(10) + 1, False), ([], [], True)),
    )
    def test_columns_same_min_vel(self, start, min_vel, expected):
        """Test that the column equals check for min_vel_strides work."""
        min_vel_cols = ["s_id", "start", "end", "gsd_id", "pre_ic", "ic", "min_vel", "tc"]
        stride_list = pd.DataFrame(columns=min_vel_cols)
        stride_list["start"] = start
        stride_list["min_vel"] = min_vel

        out = is_single_sensor_stride_list(stride_list, stride_type="min_vel")

        assert out == expected

    def test_invalid_stride_type_argument(self):
        valid_cols = ["s_id", "start", "end", "gsd_id"]
        valid = pd.DataFrame(columns=valid_cols)

        with pytest.raises(ValueError):
            is_single_sensor_stride_list(valid, stride_type="invalid_value")


class TestIsMultiSensorStrideList:
    @pytest.mark.parametrize(
        "value", (list(range(6)), "test", np.arange(6), {}, pd.DataFrame(), pd.DataFrame(columns=[*range(3)])),
    )
    def test_wrong_datatype(self, value):
        assert not is_multi_sensor_stride_list(value)

    @pytest.mark.parametrize(
        "cols, stride_types_valid",
        (
            (["s_id", "start", "end", "gsd_id"], "any"),
            (["s_id", "start", "end", "gsd_id", "something_extra"], "any"),
            (["s_id", "start", "end", "gsd_id", "pre_ic", "ic", "min_vel", "tc"], "min_vel"),
            (["s_id", "start", "end", "gsd_id", "pre_ic", "ic", "min_vel", "tc", "something_extra"], "min_vel"),
        ),
    )
    def test_valid_versions(self, cols, stride_types_valid, stride_types):
        expected_outcome = stride_types == stride_types_valid or stride_types == "any"

        out = is_multi_sensor_stride_list({"s1": pd.DataFrame(columns=cols)}, stride_type=stride_types)

        assert expected_outcome == out

    def test_only_one_invalid(self):
        valid_cols = ["s_id", "start", "end", "gsd_id"]
        invalid_cols = ["start", "end", "gsd_id"]
        valid = {"s1": pd.DataFrame(columns=valid_cols)}
        invalid = {"s2": pd.DataFrame(columns=invalid_cols), **valid}

        assert is_multi_sensor_stride_list(valid)
        assert not is_multi_sensor_stride_list(invalid)

    def test_invalid_stride_type_argument(self):
        valid_cols = ["s_id", "start", "end", "gsd_id"]
        valid = {"s1": pd.DataFrame(columns=valid_cols)}

        with pytest.raises(ValueError):
            is_multi_sensor_stride_list(valid, stride_type="invalid_value")
