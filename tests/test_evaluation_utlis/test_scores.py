import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import UndefinedMetricWarning
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

        assert_array_equal(list(eval_metrics.values()), [0.6, 0.6, 0.6])

    def test_perfect_precision_recall_f1_single(self):
        matches_df = self._create_valid_matches_df([0, 1, 2, 3, 4, 5], [], [])

        eval_metrics = precision_recall_f1_score(matches_df)

        assert_array_equal(list(eval_metrics.values()), [1.0, 1.0, 1.0])

    def test_precision_recall_f1_multi(self):
        matches_df = self._create_valid_matches_df([0, 1, 2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13])

        eval_metrics = precision_recall_f1_score({"sensor": matches_df})

        assert_array_equal(list(eval_metrics["sensor"].values()), [0.6, 0.6, 0.6])

    def test_perfect_precision_recall_f1_multi(self):
        matches_df = self._create_valid_matches_df([0, 1, 2, 3, 4, 5], [], [])

        eval_metrics = precision_recall_f1_score({"sensor": matches_df})

        assert_array_equal(list(eval_metrics["sensor"].values()), [1.0, 1.0, 1.0])


def _create_valid_matches_df(tp, fp, fn):
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


class TestDivisionByZeroReturn:
    @pytest.fixture(
        autouse=True,
        params=(
            [precision_score, [[], [], [1, 2, 3]], "warn", 0],
            [precision_score, [[], [], [1, 2, 3]], 0, 0],
            [precision_score, [[], [], [1, 2, 3]], 1, 1],
            [recall_score, [[], [1, 2, 3], []], "warn", 0],
            [recall_score, [[], [1, 2, 3], []], 0, 0],
            [recall_score, [[], [1, 2, 3], []], 1, 1],
            [f1_score, [[], [], []], "warn", 0],
            [f1_score, [[], [], []], 0, 0],
            [f1_score, [[], [], []], 1, 1],
            [precision_recall_f1_score, [[], [], []], "warn", 0],
            [precision_recall_f1_score, [[], [], []], 0, 0],
            [precision_recall_f1_score, [[], [], []], 1, 1],
        ),
    )
    def make_methods(self, request):
        self.func, self.arguments, self.zero_division, self.expected_output = request.param

    def test_division_by_zero_return(self):
        matches_df = _create_valid_matches_df(*self.arguments)

        eval_metrics = self.func(matches_df, zero_division=self.zero_division)

        assert_array_equal(
            np.array(list(eval_metrics.values()) if isinstance(eval_metrics, dict) else eval_metrics),
            self.expected_output,
        )


class TestDivisionByZeroWarnings:
    @pytest.fixture(
        autouse=True,
        params=(
            [precision_score, [[], [], [1, 2, 3]], "warn", "calculating the precision score"],
            [recall_score, [[], [1, 2, 3], []], "warn", "calculating the recall score"],
            [f1_score, [[], [], []], "warn", "calculating the f1 score"],
            [precision_recall_f1_score, [[], [], []], "warn", "calculating the f1 score"],
        ),
    )
    def make_methods(self, request):
        self.func, self.arguments, self.zero_division, self.warning_message = request.param

    def test_division_by_zero_warnings(self):
        with pytest.warns(UndefinedMetricWarning) as w:
            self.func(_create_valid_matches_df(*self.arguments), zero_division=self.zero_division)

        # check that the message matches
        assert self.warning_message in w[-1].message.args[0]


class TestDivisionByZeroError:
    @pytest.fixture(
        autouse=True,
        params=(
            [precision_score, [[], [], [1, 2, 3]], ""],
            [precision_score, [[], [], [1, 2, 3]], 2],
            [recall_score, [[], [1, 2, 3], []], ""],
            [recall_score, [[], [1, 2, 3], []], 2],
            [f1_score, [[], [], []], ""],
            [f1_score, [[], [], []], 2],
            [precision_recall_f1_score, [[], [], []], ""],
            [precision_recall_f1_score, [[], [], []], 2],
        ),
    )
    def make_methods(self, request):
        self.func, self.arguments, self.zero_division = request.param

    def test_division_by_zero_warnings(self):
        with pytest.raises(ValueError) as e:
            self.func(_create_valid_matches_df(*self.arguments), zero_division=self.zero_division)

        # check that the message matches
        assert str(e.value) == '"zero_division" must be set to "warn", 0 or 1!'
