import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from gaitmap.evaluation_utils import (
    match_stride_lists,
    evaluate_segmented_stride_list,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_f1_score,
)
from gaitmap.evaluation_utils.stride_segmentation import _get_match_type_dfs
from gaitmap.utils.exceptions import ValidationError


class TestMatchStrideList:
    def _create_valid_list(self, labels):
        df = pd.DataFrame(labels, columns=["start", "end"])
        df.index.name = "s_id"
        return df

    def test_invalid_stride_list(self):
        sl = self._create_valid_list([[0, 1], [1, 2]])

        with pytest.raises(ValidationError) as e:
            match_stride_lists([], sl)

        assert "SingleSensorStrideList" in str(e)

        with pytest.raises(ValidationError) as e:
            match_stride_lists(sl, [])

        assert "SingleSensorStrideList" in str(e)

    def test_invalid_postfix(self):
        sl = self._create_valid_list([[0, 1], [1, 2]])

        with pytest.raises(ValueError) as e:
            match_stride_lists(sl, sl, postfix_b="same", postfix_a="same")

        assert "The postfix" in str(e)

    def test_invalid_tolerance(self):
        sl = self._create_valid_list([[0, 1], [1, 2]])

        with pytest.raises(ValueError) as e:
            match_stride_lists(sl, sl, tolerance=-5)

        assert "larger 0" in str(e)

    def test_change_postfix(self):
        sl = self._create_valid_list([[0, 1], [1, 2]])

        out = match_stride_lists(sl, sl, postfix_b="_b_different", postfix_a="_a_different")

        assert_array_equal(list(out.columns), ["s_id_a_different", "s_id_b_different"])

    @pytest.mark.parametrize("one_to_one", (True, False))
    def test_simple_one_to_one_match_tolerance(self, one_to_one):
        list_left = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4]])
        list_right = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4]])

        out = match_stride_lists(list_left, list_right, tolerance=0, one_to_one=one_to_one)

        assert_array_equal(out["s_id_a"].to_numpy(), [0, 1, 2, 3])
        assert_array_equal(out["s_id_b"].to_numpy(), [0, 1, 2, 3])

    def test_simple_match_with_tolerance(self):
        list_left = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4]])
        list_left += 0.1
        list_right = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4]])

        out = match_stride_lists(list_left, list_right, tolerance=0.0, one_to_one=False)
        assert_array_equal(out["s_id_a"].to_numpy().astype(float), [0, 1, 2, 3, np.nan, np.nan, np.nan, np.nan])
        assert_array_equal(out["s_id_b"].to_numpy().astype(float), [np.nan, np.nan, np.nan, np.nan, 0, 1, 2, 3])

        out = match_stride_lists(list_left, list_right, tolerance=0.15, one_to_one=False)
        assert_array_equal(out["s_id_a"].to_numpy(), [0, 1, 2, 3])
        assert_array_equal(out["s_id_b"].to_numpy(), [0, 1, 2, 3])

    def test_simple_missing_strides_no_tolerance(self):
        list_left = self._create_valid_list([[0, 1], [2, 3], [3, 4]])
        list_right = self._create_valid_list([[0, 1], [1, 2], [2, 3]])

        out = match_stride_lists(list_left, list_right, tolerance=0)

        assert_array_equal(out["s_id_a"].to_numpy().astype(float), [0.0, 1, 2, np.nan])
        assert_array_equal(out["s_id_b"].to_numpy().astype(float), [0.0, 2, np.nan, 1])

    def test_simple_double_match_no_tolerance(self):
        list_left = self._create_valid_list([[0, 1], [1, 2], [1, 2], [2, 3], [3, 4]])
        list_right = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4], [3, 4]])

        out = match_stride_lists(list_left, list_right, tolerance=0, one_to_one=False)

        assert_array_equal(out["s_id_a"].to_numpy(), [0, 1, 2, 3, 4, 4])
        assert_array_equal(out["s_id_b"].to_numpy(), [0, 1, 1, 2, 3, 4])

    def test_simple_double_match_no_tolerance_enforce_one_to_one(self):
        list_left = self._create_valid_list([[0, 1], [1, 2], [1, 2], [2, 3], [3, 4]])
        list_right = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4], [3, 4]])

        out = match_stride_lists(list_left, list_right, tolerance=0, one_to_one=True)

        assert_array_equal(out["s_id_a"].to_numpy().astype(float), [0, 1, 2, 3, 4, np.nan])
        assert_array_equal(out["s_id_b"].to_numpy().astype(float), [0, 1, np.nan, 2, 3, 4])

    def test_double_match_with_tolerance_enforce_one_to_one(self):
        list_left = self._create_valid_list([[0, 1], [1.1, 2.1], [1, 2], [2, 3], [3, 4]])
        list_right = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3.1, 3.9], [3, 4]])

        out = match_stride_lists(list_left, list_right, tolerance=0.15, one_to_one=True)

        assert_array_equal(out["s_id_a"].to_numpy().astype(float), [0, 1, 2, 3, 4, np.nan])
        assert_array_equal(out["s_id_b"].to_numpy().astype(float), [0, np.nan, 1, 2, 4, 3])

    def test_one_sided_double_match_no_tolerance_enforce_one_to_one(self):
        list_left = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4], [1, 2]])
        list_right = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4]])

        out = match_stride_lists(list_left, list_right, tolerance=0, one_to_one=True)

        assert_array_equal(out["s_id_a"].to_numpy().astype(float), [0, 1, 2, 3, 4])
        assert_array_equal(out["s_id_b"].to_numpy().astype(float), [0, 1, 2, 3, np.nan])

    @pytest.mark.parametrize("side", ("a", "b"))
    def test_empty_stride_lists(self, side):
        opposite = [s for s in ("a", "b") if s != side][0]
        sl = self._create_valid_list([[0, 1], [1, 2], [2, 3]])
        empty = self._create_valid_list([])

        emtpy_name = "stride_list_" + side
        normal_name = "stride_list_" + opposite

        out = match_stride_lists(**{emtpy_name: empty, normal_name: sl})

        assert_array_equal(out["s_id_" + side].to_numpy().astype(float), [np.nan, np.nan, np.nan])
        assert_array_equal(out["s_id_" + opposite].to_numpy().astype(float), [0.0, 1, 2])

    def test_empty_stride_lists_both(self):
        empty = self._create_valid_list([])

        out = match_stride_lists(empty, empty)

        assert len(out) == 0


class TestEvaluateSegmentedStrideList:
    def _create_valid_list(self, labels):
        df = pd.DataFrame(labels, columns=["start", "end"])
        df.index.name = "s_id"
        return df

    def test_segmented_stride_list_perfect_match(self):
        list_ground_truth = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4]])
        list_predicted = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4]])
        matches = evaluate_segmented_stride_list(list_ground_truth, list_predicted)

        assert np.all(matches["match_type"] == "tp")

    def test_segmented_stride_list_empty_ground_truth(self):
        list_ground_truth = self._create_valid_list([])
        list_predicted = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4]])
        matches = evaluate_segmented_stride_list(list_ground_truth, list_predicted)

        matches = _get_match_type_dfs(matches)

        assert matches["tp"].empty
        assert matches["fn"].empty
        assert len(matches["fp"]) == len(list_predicted)
        assert_array_equal(matches["fp"]["s_id"].to_numpy(), [0, 1, 2, 3])
        assert_array_equal(
            matches["fp"]["s_id_ground_truth"].to_numpy().astype(float), np.array([np.nan, np.nan, np.nan, np.nan])
        )
        assert len(list_ground_truth) == (len(matches["tp"]) + len(matches["fn"]))

    def test_segmented_stride_list_empty_prediction(self):
        list_ground_truth = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4]])
        list_predicted = self._create_valid_list([])

        matches = evaluate_segmented_stride_list(list_ground_truth, list_predicted)

        matches = _get_match_type_dfs(matches)

        assert matches["tp"].empty
        assert matches["fp"].empty
        assert len(matches["fn"]) == len(list_ground_truth)
        assert_array_equal(matches["fn"]["s_id_ground_truth"].to_numpy(), [0, 1, 2, 3])
        assert_array_equal(matches["fn"]["s_id"].to_numpy().astype(float), [np.nan, np.nan, np.nan, np.nan])
        assert len(list_ground_truth) == (len(matches["tp"]) + len(matches["fn"]))

    def test_segmented_stride_list_match(self):
        list_ground_truth = self._create_valid_list([[20, 30], [30, 40], [40, 50], [50, 60]])
        list_predicted = self._create_valid_list([[0, 10], [11, 19], [19, 30], [30, 41], [70, 80], [80, 90]])

        matches = evaluate_segmented_stride_list(list_ground_truth, list_predicted, tolerance=1)

        matches = _get_match_type_dfs(matches)

        assert_array_equal(matches["tp"]["s_id"].to_numpy(), [2, 3])
        assert_array_equal(matches["tp"]["s_id_ground_truth"].to_numpy(), [0, 1])

        assert_array_equal(matches["fp"]["s_id"].to_numpy(), [0, 1, 4, 5])
        assert_array_equal(
            matches["fp"]["s_id_ground_truth"].to_numpy().astype(float), np.array([np.nan, np.nan, np.nan, np.nan])
        )

        assert_array_equal(matches["fn"]["s_id"].to_numpy().astype(float), [np.nan, np.nan])
        assert_array_equal(matches["fn"]["s_id_ground_truth"].to_numpy(), [2, 3])

        assert len(list_ground_truth) == (len(matches["tp"]) + len(matches["fn"]))

    def test_segmented_stride_list_no_match(self):
        list_ground_truth = self._create_valid_list([[20, 30], [30, 40], [40, 50]])
        list_predicted = self._create_valid_list([[60, 70], [70, 80], [90, 100]])

        matches = evaluate_segmented_stride_list(list_ground_truth, list_predicted, tolerance=0)

        matches = _get_match_type_dfs(matches)

        assert matches["tp"].empty

        assert_array_equal(matches["fn"]["s_id"].to_numpy().astype(float), np.array([np.nan, np.nan, np.nan]))
        assert_array_equal(matches["fn"]["s_id_ground_truth"].to_numpy(), [0, 1, 2])

        assert_array_equal(matches["fp"]["s_id"].to_numpy(), [0, 1, 2])
        assert_array_equal(
            matches["fp"]["s_id_ground_truth"].to_numpy().astype(float), np.array([np.nan, np.nan, np.nan])
        )

        assert len(list_ground_truth) == (len(matches["tp"]) + len(matches["fn"]))

    def test_segmented_stride_list_double_match_predicted_many_to_one(self):
        list_ground_truth = self._create_valid_list([[20, 30]])
        list_predicted = self._create_valid_list([[18, 30], [20, 28]])

        matches = evaluate_segmented_stride_list(list_ground_truth, list_predicted, tolerance=2, one_to_one=False)

        matches = _get_match_type_dfs(matches)

        assert_array_equal(matches["tp"]["s_id"].to_numpy(), [0, 1])
        assert_array_equal(matches["tp"]["s_id_ground_truth"].to_numpy(), [0, 0])

        assert matches["fp"].empty
        assert matches["fn"].empty

        assert len(list_ground_truth) != (len(matches["tp"]) + len(matches["fn"]))

    def test_segmented_stride_list_double_match_predicted_one_to_one(self):
        list_ground_truth = self._create_valid_list([[20, 30]])
        list_predicted = self._create_valid_list([[18, 30], [20, 28]])

        matches = evaluate_segmented_stride_list(list_ground_truth, list_predicted, tolerance=2, one_to_one=True)

        matches = _get_match_type_dfs(matches)

        assert_array_equal(matches["tp"]["s_id"].to_numpy(), 0)
        assert_array_equal(matches["tp"]["s_id_ground_truth"].to_numpy(), [0])

        assert_array_equal(matches["fp"]["s_id"].to_numpy(), 1)
        assert_array_equal(matches["fp"]["s_id_ground_truth"].to_numpy().astype(float), np.array(np.nan))

        assert matches["fn"].empty

        assert len(list_ground_truth) == (len(matches["tp"]) + len(matches["fn"]))


class TestEvaluationScores:
    def _create_valid_matches_df(self, tp, fp, fn):
        tp_df = pd.DataFrame(
            np.column_stack([tp, tp, np.repeat("tp", len(tp))]), columns=["s_id", "s_id_ground_truth", "match_type"]
        )
        fp_df = pd.DataFrame(
            np.column_stack([fp, np.repeat(np.nan, len(fp)), np.repeat("fp", len(fp))]),
            columns=["s_id", "s_id_ground_truth", "match_type"],
        )
        fn_df = pd.DataFrame(
            np.column_stack([np.repeat(np.nan, len(fn)), fn, np.repeat("fn", len(fn))]),
            columns=["s_id", "s_id_ground_truth", "match_type"],
        )

        return pd.concat([tp_df, fp_df, fn_df])

    def test_precision(self):
        matches_df = self._create_valid_matches_df([0, 1, 2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13])

        precision = precision_score(matches_df)

        assert_array_equal(precision, 0.6)

    def test_perfect_precision(self):
        matches_df = self._create_valid_matches_df([0, 1, 2, 3, 4, 5], [], [10, 11, 12, 13])

        precision = precision_score(matches_df)

        assert_array_equal(precision, 1.0)

    def test_recall(self):
        matches_df = self._create_valid_matches_df([0, 1, 2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13])

        recall = recall_score(matches_df)

        assert_array_equal(recall, 0.6)

    def test_perfect_recall(self):
        matches_df = self._create_valid_matches_df([0, 1, 2, 3, 4, 5], [6, 7, 8, 9], [])

        recall = recall_score(matches_df)

        assert_array_equal(recall, 1.0)

    def test_f1_score(self):
        matches_df = self._create_valid_matches_df([0, 1, 2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13])

        f1 = f1_score(matches_df)

        assert_array_equal(f1, 0.6)

    def test_perfect_f1_score(self):
        matches_df = self._create_valid_matches_df([0, 1, 2, 3, 4, 5], [], [])

        f1 = f1_score(matches_df)

        assert_array_equal(f1, 1.0)

    def test_precision_recall_f1(self):
        matches_df = self._create_valid_matches_df([0, 1, 2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13])

        eval_metrics = precision_recall_f1_score(matches_df)

        assert_array_equal(eval_metrics, [0.6, 0.6, 0.6])

    def test_perfect_precision_recall_f1(self):
        matches_df = self._create_valid_matches_df([0, 1, 2, 3, 4, 5], [], [])

        eval_metrics = precision_recall_f1_score(matches_df)

        assert_array_equal(eval_metrics, [1.0, 1.0, 1.0])
