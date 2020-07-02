import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from gaitmap.evaluation_utils import match_stride_lists


class TestMatchStrideList:
    def _create_valid_list(self, labels):
        df = pd.DataFrame(labels, columns=["start", "end"])
        df.index.name = "s_id"
        return df

    def test_invalid_stride_list(self):
        sl = self._create_valid_list([[0, 1], [1, 2]])

        with pytest.raises(ValueError) as e:
            match_stride_lists([], sl)

        assert "stride_list_left" in str(e)

        with pytest.raises(ValueError) as e:
            match_stride_lists(sl, [])

        assert "stride_list_right" in str(e)

    def test_invalid_postfix(self):
        sl = self._create_valid_list([[0, 1], [1, 2]])

        with pytest.raises(ValueError) as e:
            match_stride_lists(sl, sl, right_postfix="same", left_postfix="same")

        assert "The postfix" in str(e)

    def test_invalid_tolerance(self):
        sl = self._create_valid_list([[0, 1], [1, 2]])

        with pytest.raises(ValueError) as e:
            match_stride_lists(sl, sl, tolerance=-5)

        assert "larger 0" in str(e)

    def test_change_postfix(self):
        sl = self._create_valid_list([[0, 1], [1, 2]])

        out = match_stride_lists(sl, sl, right_postfix="_right_different", left_postfix="_left_different")

        assert_array_equal(list(out.columns), ["s_id_left_different", "s_id_right_different"])

    @pytest.mark.parametrize("one_to_one", (True, False))
    def test_simple_one_to_one_match_tolerance(self, one_to_one):
        list_left = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4]])
        list_right = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4]])

        out = match_stride_lists(list_left, list_right, tolerance=0, one_to_one=one_to_one)

        assert_array_equal(out["s_id_left"].to_numpy(), [0, 1, 2, 3])
        assert_array_equal(out["s_id_right"].to_numpy(), [0, 1, 2, 3])

    def test_simple_match_with_tolerance(self):
        list_left = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4]])
        list_left += 0.1
        list_right = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4]])

        out = match_stride_lists(list_left, list_right, tolerance=0.0, one_to_one=False)
        assert_array_equal(out["s_id_left"].to_numpy().astype(float), [0, 1, 2, 3, np.nan, np.nan, np.nan, np.nan])
        assert_array_equal(out["s_id_right"].to_numpy().astype(float), [np.nan, np.nan, np.nan, np.nan, 0, 1, 2, 3])

        out = match_stride_lists(list_left, list_right, tolerance=0.15, one_to_one=False)
        assert_array_equal(out["s_id_left"].to_numpy(), [0, 1, 2, 3])
        assert_array_equal(out["s_id_right"].to_numpy(), [0, 1, 2, 3])

    def test_simple_missing_strides_no_tolerance(self):
        list_left = self._create_valid_list([[0, 1], [2, 3], [3, 4]])
        list_right = self._create_valid_list([[0, 1], [1, 2], [2, 3]])

        out = match_stride_lists(list_left, list_right, tolerance=0)

        assert_array_equal(out["s_id_left"].to_numpy().astype(float), [0.0, 1, 2, np.nan])
        assert_array_equal(out["s_id_right"].to_numpy().astype(float), [0.0, 2, np.nan, 1])

    def test_simple_double_match_no_tolerance(self):
        list_left = self._create_valid_list([[0, 1], [1, 2], [1, 2], [2, 3], [3, 4]])
        list_right = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4], [3, 4]])

        out = match_stride_lists(list_left, list_right, tolerance=0, one_to_one=False)

        assert_array_equal(out["s_id_left"].to_numpy(), [0, 1, 2, 3, 4, 4])
        assert_array_equal(out["s_id_right"].to_numpy(), [0, 1, 1, 2, 3, 4])

    def test_simple_double_match_no_tolerance_enforce_one_to_one(self):
        list_left = self._create_valid_list([[0, 1], [1, 2], [1, 2], [2, 3], [3, 4]])
        list_right = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4], [3, 4]])

        out = match_stride_lists(list_left, list_right, tolerance=0, one_to_one=True)

        assert_array_equal(out["s_id_left"].to_numpy().astype(float), [0, 1, 2, 3, 4, np.nan])
        assert_array_equal(out["s_id_right"].to_numpy().astype(float), [0, 1, np.nan, 2, 3, 4])

    def test_double_match_with_tolerance_enforce_one_to_one(self):
        list_left = self._create_valid_list([[0, 1], [1.1, 2.1], [1, 2], [2, 3], [3, 4]])
        list_right = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3.1, 3.9], [3, 4]])

        out = match_stride_lists(list_left, list_right, tolerance=0.15, one_to_one=True)

        assert_array_equal(out["s_id_left"].to_numpy().astype(float), [0, 1, 2, 3, 4, np.nan])
        assert_array_equal(out["s_id_right"].to_numpy().astype(float), [0, np.nan, 1, 2, 4, 3])

    def test_one_sided_double_match_no_tolerance_enforce_one_to_one(self):
        list_left = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4], [1, 2]])
        list_right = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4]])

        out = match_stride_lists(list_left, list_right, tolerance=0, one_to_one=True)

        assert_array_equal(out["s_id_left"].to_numpy().astype(float), [0, 1, 2, 3, 4])
        assert_array_equal(out["s_id_right"].to_numpy().astype(float), [0, 1, 2, 3, np.nan])
