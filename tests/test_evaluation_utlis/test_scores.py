import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from gaitmap.evaluation_utils.scores import precision_score, recall_score, f1_score, precision_recall_f1_score


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

    def test_precision_single(self):
        matches_df = self._create_valid_matches_df([0, 1, 2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13])

        precision = precision_score(matches_df)

        assert_array_equal(precision, 0.6)

    def test_perfect_precision_single(self):
        matches_df = self._create_valid_matches_df([0, 1, 2, 3, 4, 5], [], [10, 11, 12, 13])

        precision = precision_score(matches_df)

        assert_array_equal(precision, 1.0)

    def test_precision_multi(self):
        matches_df = self._create_valid_matches_df([0, 1, 2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13])

        precision = precision_score({"sensor": matches_df})

        assert_array_equal(precision["sensor"], 0.6)

    def test_perfect_precision_multi(self):
        matches_df = self._create_valid_matches_df([0, 1, 2, 3, 4, 5], [], [10, 11, 12, 13])

        precision = precision_score({"sensor": matches_df})

        assert_array_equal(precision["sensor"], 1.0)

    def test_recall_single(self):
        matches_df = self._create_valid_matches_df([0, 1, 2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13])

        recall = recall_score(matches_df)

        assert_array_equal(recall, 0.6)

    def test_perfect_recall_single(self):
        matches_df = self._create_valid_matches_df([0, 1, 2, 3, 4, 5], [6, 7, 8, 9], [])

        recall = recall_score(matches_df)

        assert_array_equal(recall, 1.0)

    def test_recall_multi(self):
        matches_df = self._create_valid_matches_df([0, 1, 2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13])

        recall = recall_score({"sensor": matches_df})

        assert_array_equal(recall["sensor"], 0.6)

    def test_perfect_recall_multi(self):
        matches_df = self._create_valid_matches_df([0, 1, 2, 3, 4, 5], [6, 7, 8, 9], [])

        recall = recall_score({"sensor": matches_df})

        assert_array_equal(recall["sensor"], 1.0)

    def test_f1_score_single(self):
        matches_df = self._create_valid_matches_df([0, 1, 2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13])

        f1 = f1_score(matches_df)

        assert_array_equal(f1, 0.6)

    def test_perfect_f1_score_single(self):
        matches_df = self._create_valid_matches_df([0, 1, 2, 3, 4, 5], [], [])

        f1 = f1_score(matches_df)

        assert_array_equal(f1, 1.0)

    def test_f1_score_multi(self):
        matches_df = self._create_valid_matches_df([0, 1, 2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13])

        f1 = f1_score({"sensor": matches_df})

        assert_array_equal(f1["sensor"], 0.6)

    def test_perfect_f1_score_multi(self):
        matches_df = self._create_valid_matches_df([0, 1, 2, 3, 4, 5], [], [])

        f1 = f1_score({"sensor": matches_df})

        assert_array_equal(f1["sensor"], 1.0)

    def test_precision_recall_f1_single(self):
        matches_df = self._create_valid_matches_df([0, 1, 2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13])

        eval_metrics = precision_recall_f1_score(matches_df)

        assert_array_equal(eval_metrics, [0.6, 0.6, 0.6])

    def test_perfect_precision_recall_f1_single(self):
        matches_df = self._create_valid_matches_df([0, 1, 2, 3, 4, 5], [], [])

        eval_metrics = precision_recall_f1_score(matches_df)

        assert_array_equal(eval_metrics, [1.0, 1.0, 1.0])

    def test_precision_recall_f1_multi(self):
        matches_df = self._create_valid_matches_df([0, 1, 2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13])

        eval_metrics = precision_recall_f1_score({"sensor": matches_df})

        assert_array_equal(eval_metrics["sensor"], [0.6, 0.6, 0.6])

    def test_perfect_precision_recall_f1_multi(self):
        matches_df = self._create_valid_matches_df([0, 1, 2, 3, 4, 5], [], [])

        eval_metrics = precision_recall_f1_score({"sensor": matches_df})

        assert_array_equal(eval_metrics["sensor"], [1.0, 1.0, 1.0])
