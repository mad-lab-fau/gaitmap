import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from gaitmap.evaluation_utils import evaluate_stride_event_list

from gaitmap.evaluation_utils.scores import _get_match_type_dfs


class TestEvaluateStrideEventList:
    def _create_valid_list(self, labels, extra_columns=None):
        columns = ["start", "end"]

        if extra_columns:
            columns += [extra_columns] if isinstance(extra_columns, str) else extra_columns

        df = pd.DataFrame(labels, columns=columns)
        df.index.name = "s_id"

        return df

    def test_too_many_column_names(self):
        sl = self._create_valid_list([[0, 1, 10], [1, 2, 20]], "ic")

        with pytest.raises(ValueError) as e:
            evaluate_stride_event_list(sl, sl, ["1", "2", "3"])

        assert "match_cols needs to be one" in str(e.value)

    def test_invalid_column_name(self):
        sl = self._create_valid_list([[0, 1, 10], [1, 2, 20]], "ic")

        with pytest.raises(ValueError) as e:
            evaluate_stride_event_list(sl, sl, "wrong_column")

        assert "match_cols needs to be one" in str(e.value)

    def test_no_column_name(self):
        sl = self._create_valid_list([[0, 1, 10], [1, 2, 20]], "ic")

        with pytest.raises(ValueError) as e:
            evaluate_stride_event_list(sl, sl, "")

        assert "match_cols can not be none" in str(e.value)

    def test_perfect_match(self):
        sl = self._create_valid_list([[0, 1, 10], [1, 2, 20], [2, 3, 30]], "ic")

        out = evaluate_stride_event_list(sl, sl, "ic", tolerance=0)
        out = _get_match_type_dfs(out)

        assert_array_equal(out["tp"]["s_id"].to_numpy(), [0, 1, 2])
        assert_array_equal(out["tp"]["s_id_ground_truth"].to_numpy(), [0, 1, 2])
        assert len(out["fp"]) == 0
        assert len(out["fn"]) == 0
        assert len(out) == (len(out["tp"]) + len(out["fn"]))

    def test_match(self):
        sl1 = self._create_valid_list([[0, 1, 0], [1, 2, 20], [2, 3, 30]], "ic")
        sl2 = self._create_valid_list([[0, 1, 10], [1, 2, 20], [2, 3, 30]], "ic")

        out = evaluate_stride_event_list(sl1, sl2, "ic", tolerance=0)
        out = _get_match_type_dfs(out)

        assert_array_equal(out["tp"]["s_id"].to_numpy(), [1, 2])
        assert_array_equal(out["tp"]["s_id_ground_truth"].to_numpy(), [1, 2])
        assert_array_equal(out["fp"]["s_id"].to_numpy(), [0])
        assert_array_equal(out["fp"]["s_id_ground_truth"].to_numpy().astype(np.float), [np.nan])
        assert_array_equal(out["fn"]["s_id"].to_numpy().astype(np.float), [np.nan])
        assert_array_equal(out["fn"]["s_id_ground_truth"].to_numpy(), [0])
        assert len(out) == (len(out["tp"]) + len(out["fn"]))
