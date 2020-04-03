"""Test the dataset helpers."""
from collections import Callable
from typing import List

import pytest

from gaitmap.utils.consts import SF_COLS, SF_GYR, SF_ACC, BF_COLS, BF_GYR, BF_ACC
from gaitmap.utils.dataset_helper import _has_sf_cols, _has_bf_cols


class TestColumnHelper:

    method: Callable
    cols: List[str]
    gyr_cols: List[str]
    acc_cols: List[str]

    @pytest.fixture(params=['sf', 'bf'], autouse=True)
    def select_method(self, request):
        self.method = {'sf': _has_sf_cols, 'bf': _has_bf_cols}[request.param]
        self.cols = {'sf': SF_COLS, 'bf': BF_COLS}[request.param]
        self.gyr_cols = {'sf': SF_GYR, 'bf': BF_GYR}[request.param]
        self.acc_cols = {'sf': SF_ACC, 'bf': BF_ACC}[request.param]

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

