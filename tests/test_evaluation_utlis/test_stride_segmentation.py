import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from gaitmap.evaluation_utils import evaluate_segmented_stride_list, match_stride_lists
from gaitmap.evaluation_utils.scores import _get_match_type_dfs
from gaitmap.evaluation_utils.stride_segmentation import _match_label_lists
from gaitmap.utils.exceptions import ValidationError


class TestMatchStrideList:
    def _create_valid_list(self, labels):
        df = pd.DataFrame(labels, columns=["start", "end"])
        df.index.name = "s_id"
        return df

    def test_invalid_stride_list(self):
        sl = self._create_valid_list([[0, 1], [1, 2]])

        with pytest.raises(ValidationError) as e:
            match_stride_lists(stride_list_a={}, stride_list_b=sl)

        assert "SingleSensorStrideList" in str(e.value)

        with pytest.raises(ValidationError) as e:
            match_stride_lists(stride_list_a=sl, stride_list_b={})

        assert "SingleSensorStrideList" in str(e.value)

    def test_invalid_postfix(self):
        sl = self._create_valid_list([[0, 1], [1, 2]])

        with pytest.raises(ValueError) as e:
            match_stride_lists(stride_list_a=sl, stride_list_b=sl, postfix_a="same", postfix_b="same")

        assert "The postfix" in str(e)

    def test_invalid_tolerance(self):
        sl = self._create_valid_list([[0, 1], [1, 2]])

        with pytest.raises(ValueError) as e:
            match_stride_lists(stride_list_a=sl, stride_list_b=sl, tolerance=-5)

        assert "larger 0" in str(e)

    def test_change_postfix(self):
        sl = self._create_valid_list([[0, 1], [1, 2]])

        out = match_stride_lists(stride_list_a=sl, stride_list_b=sl, postfix_a="_a_different", postfix_b="_b_different")

        assert_array_equal(list(out.columns), ["s_id_a_different", "s_id_b_different"])

    @pytest.mark.parametrize("one_to_one", (True, False))
    def test_simple_one_to_one_match_tolerance(self, one_to_one):
        list_left = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4]])
        list_right = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4]])

        out = match_stride_lists(stride_list_a=list_left, stride_list_b=list_right, tolerance=0, one_to_one=one_to_one)

        assert_array_equal(out["s_id_a"].to_numpy(), [0, 1, 2, 3])
        assert_array_equal(out["s_id_b"].to_numpy(), [0, 1, 2, 3])

    def test_simple_match_with_tolerance(self):
        list_left = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4]])
        list_left += 0.1
        list_right = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4]])

        out = match_stride_lists(stride_list_a=list_left, stride_list_b=list_right, tolerance=0.0, one_to_one=False)
        assert_array_equal(out["s_id_a"].to_numpy().astype(float), [0, 1, 2, 3, np.nan, np.nan, np.nan, np.nan])
        assert_array_equal(out["s_id_b"].to_numpy().astype(float), [np.nan, np.nan, np.nan, np.nan, 0, 1, 2, 3])

        out = match_stride_lists(stride_list_a=list_left, stride_list_b=list_right, tolerance=0.15, one_to_one=False)
        assert_array_equal(out["s_id_a"].to_numpy(), [0, 1, 2, 3])
        assert_array_equal(out["s_id_b"].to_numpy(), [0, 1, 2, 3])

    def test_simple_missing_strides_no_tolerance(self):
        list_left = self._create_valid_list([[0, 1], [2, 3], [3, 4]])
        list_right = self._create_valid_list([[0, 1], [1, 2], [2, 3]])

        out = match_stride_lists(stride_list_a=list_left, stride_list_b=list_right, tolerance=0)

        assert_array_equal(out["s_id_a"].to_numpy().astype(float), [0.0, 1, 2, np.nan])
        assert_array_equal(out["s_id_b"].to_numpy().astype(float), [0.0, 2, np.nan, 1])

    def test_simple_double_match_no_tolerance(self):
        list_left = self._create_valid_list([[0, 1], [1, 2], [1, 2], [2, 3], [3, 4]])
        list_right = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4], [3, 4]])

        out = match_stride_lists(stride_list_a=list_left, stride_list_b=list_right, tolerance=0, one_to_one=False)

        assert_array_equal(out["s_id_a"].to_numpy(), [0, 1, 2, 3, 4, 4])
        assert_array_equal(out["s_id_b"].to_numpy(), [0, 1, 1, 2, 3, 4])

    def test_simple_double_match_no_tolerance_enforce_one_to_one(self):
        list_left = self._create_valid_list([[0, 1], [1, 2], [1, 2], [2, 3], [3, 4]])
        list_right = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4], [3, 4]])

        out = match_stride_lists(stride_list_a=list_left, stride_list_b=list_right, tolerance=0, one_to_one=True)

        assert_array_equal(out["s_id_a"].to_numpy().astype(float), [0, 1, 2, 3, 4, np.nan])
        assert_array_equal(out["s_id_b"].to_numpy().astype(float), [0, 1, np.nan, 2, 3, 4])

    def test_double_match_with_tolerance_enforce_one_to_one(self):
        list_left = self._create_valid_list([[0, 1], [1.1, 2.1], [1, 2], [2, 3], [3, 4]])
        list_right = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3.1, 3.9], [3, 4]])

        out = match_stride_lists(stride_list_a=list_left, stride_list_b=list_right, tolerance=0.15, one_to_one=True)

        assert_array_equal(out["s_id_a"].to_numpy().astype(float), [0, 1, 2, 3, 4, np.nan])
        assert_array_equal(out["s_id_b"].to_numpy().astype(float), [0, np.nan, 1, 2, 4, 3])

    def test_one_sided_double_match_no_tolerance_enforce_one_to_one(self):
        list_left = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4], [1, 2]])
        list_right = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4]])

        out = match_stride_lists(stride_list_a=list_left, stride_list_b=list_right, tolerance=0, one_to_one=True)

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

        out = match_stride_lists(stride_list_a=empty, stride_list_b=empty)

        assert len(out) == 0

    def test_multi_stride_lists_no_tolerance(self):
        stride_list_left_a = self._create_valid_list([[0, 1], [2, 3], [4, 5], [6, 7]])
        stride_list_right_a = self._create_valid_list([[1, 2], [3, 4], [5, 6]])
        multi_stride_list_a = {"left_sensor": stride_list_left_a, "right_sensor": stride_list_right_a}

        stride_list_left_b = self._create_valid_list([[0, 1], [4, 5], [6, 7], [8, 9]])
        stride_list_right_b = self._create_valid_list([[3, 4], [1, 2], [5, 6]])
        multi_stride_list_b = {"left_sensor": stride_list_left_b, "right_sensor": stride_list_right_b}

        out = match_stride_lists(stride_list_a=multi_stride_list_a, stride_list_b=multi_stride_list_b, tolerance=0)

        assert_array_equal(out["left_sensor"]["s_id_a"].to_numpy().astype(float), [0, 1, 2, 3, np.nan])
        assert_array_equal(out["left_sensor"]["s_id_b"].to_numpy().astype(float), [0, np.nan, 1, 2, 3])

        assert_array_equal(out["right_sensor"]["s_id_a"].to_numpy().astype(float), [0, 1, 2])
        assert_array_equal(out["right_sensor"]["s_id_b"].to_numpy().astype(float), [1, 0, 2])

    def test_multi_stride_lists_with_tolerance(self):
        stride_list_left_a = self._create_valid_list([[0, 1], [2, 3], [4, 5], [6, 7]])
        stride_list_right_a = self._create_valid_list([[1, 2], [3, 4], [5, 6]])
        multi_stride_list_a = {"left_sensor": stride_list_left_a, "right_sensor": stride_list_right_a}

        stride_list_left_b = self._create_valid_list([[0, 2], [2, 4], [4, 6], [6, 9]])
        stride_list_right_b = self._create_valid_list([[0, 2], [2, 4], [3, 5]])
        multi_stride_list_b = {"left_sensor": stride_list_left_b, "right_sensor": stride_list_right_b}

        out = match_stride_lists(stride_list_a=multi_stride_list_a, stride_list_b=multi_stride_list_b, tolerance=1)

        assert_array_equal(out["left_sensor"]["s_id_a"].to_numpy().astype(float), [0, 1, 2, 3, np.nan])
        assert_array_equal(out["left_sensor"]["s_id_b"].to_numpy().astype(float), [0, 1, 2, np.nan, 3])

        assert_array_equal(out["right_sensor"]["s_id_a"].to_numpy().astype(float), [0, 1, 2, np.nan])
        assert_array_equal(out["right_sensor"]["s_id_b"].to_numpy().astype(float), [0, 1, np.nan, 2])

    def test_empty_multi_stride_lists_both(self):
        empty = self._create_valid_list([])

        out = match_stride_lists(stride_list_a={"left": empty}, stride_list_b={"left": empty})

        for dataframe in out.values():
            assert dataframe.empty

    def test_empty_multi_stride_lists(self):
        full = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4]])
        empty = self._create_valid_list([])

        out = match_stride_lists(stride_list_a={"left": full}, stride_list_b={"left": empty})

        assert_array_equal(out["left"]["s_id_a"].to_numpy().astype(float), [0, 1, 2, 3])
        assert_array_equal(out["left"]["s_id_b"].to_numpy().astype(float), [np.nan, np.nan, np.nan, np.nan])

        out = match_stride_lists(stride_list_a={"left": empty}, stride_list_b={"left": full})

        assert_array_equal(out["left"]["s_id_a"].to_numpy().astype(float), [np.nan, np.nan, np.nan, np.nan])
        assert_array_equal(out["left"]["s_id_b"].to_numpy().astype(float), [0, 1, 2, 3])

        with pytest.raises(ValidationError) as e:
            match_stride_lists(stride_list_a={}, stride_list_b={})

        assert "object does not contain any data/contains no sensors" in str(e.value)

    def test_one_multi_one_single_list(self):
        multi = {"sensor": self._create_valid_list([[0, 1], [2, 3], [4, 5], [6, 7]])}
        single = self._create_valid_list([[1, 2], [3, 4], [5, 6]])

        with pytest.raises(ValidationError) as e:
            match_stride_lists(stride_list_a=multi, stride_list_b=single)

        assert "not of same type" in str(e)

        with pytest.raises(ValidationError) as e:
            match_stride_lists(stride_list_a=single, stride_list_b=multi)

        assert "not of same type" in str(e)

    def test_no_common_sensors_multi_stride_lists(self):
        full = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4]])

        with pytest.raises(ValidationError) as e:
            match_stride_lists(stride_list_a={"left": full}, stride_list_b={"right": full})

        assert "do not have any common sensors" in str(e)

    def test_some_common_sensors_multi_stride_lists(self):
        stride_list_left_a = self._create_valid_list([[0, 1], [2, 3], [4, 5], [6, 7]])
        stride_list_right_a = self._create_valid_list([[1, 2], [3, 4], [5, 6]])
        multi_stride_list_a = {"left_sensor": stride_list_left_a, "right_sensor": stride_list_right_a}

        stride_list_left_b = self._create_valid_list([[0, 1], [4, 5], [6, 7], [8, 9]])
        stride_list_right_b = self._create_valid_list([[3, 4], [1, 2], [5, 6]])
        multi_stride_list_b = {
            "left_sensor": stride_list_left_b,
            "right_sensor": stride_list_right_b,
            "wrong_sensor": stride_list_right_a,
        }

        with pytest.raises(KeyError):
            match_stride_lists(stride_list_a=multi_stride_list_a, stride_list_b=multi_stride_list_b, tolerance=0)[
                "wrong_sensor"
            ]


class TestSpecialMatchStrideList:
    def _create_valid_list(self, labels, extra_columns=None):
        if extra_columns:
            if isinstance(extra_columns, str):
                df = pd.DataFrame(labels, columns=["start", "end", extra_columns])
            else:
                columns = ["start", "end"]
                for extra_column in extra_columns:
                    columns.append(extra_column)

                df = pd.DataFrame(labels, columns=columns)
        else:
            df = pd.DataFrame(labels, columns=["start", "end"])
        df.index.name = "s_id"
        return df

    @pytest.mark.parametrize("value", ("wrong_column", ["1", "2", "3"]))
    def test_invalid_match_cols(self, value):
        sl = self._create_valid_list([[0, 1, 10], [1, 2, 20]], "ic")

        with pytest.raises(ValueError) as e:
            match_stride_lists(stride_list_a=sl, stride_list_b=sl, match_cols=value)

        assert "One or more selected columns" in str(e.value)
        assert str(value) in str(e.value)

    def test_perfect_match(self):
        sl = self._create_valid_list([[0, 1, 10], [1, 2, 20], [2, 3, 30]], "ic")

        out = match_stride_lists(stride_list_a=sl, stride_list_b=sl, match_cols="ic", tolerance=0)

        assert_array_equal(out["s_id_a"].to_numpy(), [0, 1, 2])
        assert_array_equal(out["s_id_b"].to_numpy(), [0, 1, 2])

    def test_match(self):
        sl1 = self._create_valid_list([[0, 1, 0], [1, 2, 20], [2, 3, 30]], "ic")
        sl2 = self._create_valid_list([[0, 1, 10], [1, 2, 20], [2, 3, 30]], "ic")

        out = match_stride_lists(stride_list_a=sl1, stride_list_b=sl2, match_cols="ic", tolerance=0)

        assert_array_equal(out["s_id_a"].to_numpy().astype(float), [0, 1, 2, np.nan])
        assert_array_equal(out["s_id_b"].to_numpy().astype(float), [np.nan, 1, 2, 0])

    @pytest.mark.parametrize(
        ("input_param", "ground_truth", "tolerance", "one_to_one", "expectation"),
        [
            (
                np.array([[0, 1, 2], [10, 20, 30], [30, 40, 50]]),
                np.array([[0, 1, 2], [10, 20, 30], [30, 40, 50]]),
                0,
                True,
                [[0, 1, 2], [0, 1, 2]],
            ),
            (
                np.array([[0, 1, 2], [10, 20, 30], [30, 40, 50]]),
                np.array([[30, 40, 50], [10, 20, 30], [0, 1, 2]]),
                0,
                True,
                [[0, 1, 2], [2, 1, 0]],
            ),
            (
                np.array([[0, 1, 2, 3], [10, 20, 30, 40], [101, 200, 300, 400]]),
                np.array([[0, 1, 2, 4], [101, 201, 300, 400], [10, 20, 31, 40]]),
                1,
                True,
                [[0, 1, 2], [0, 2, 1]],
            ),
            (
                np.array([[0, 1, 2, 3], [10, 20, 30, 40], [101, 200, 300, 400], [0, 1, 2, 3]]),
                np.array([[0, 1, 2, 4], [101, 201, 300, 400], [10, 20, 31, 40], [0, 1, 2, 4]]),
                1,
                False,
                [[0, 0, 1, 2, 3, 3], [0, 3, 2, 1, 0, 3]],
            ),
            (
                np.array([[1], [2], [3], [4], [5]]),
                np.array([[6], [7], [8], [9], [10]]),
                0,
                True,
                [[], []],
            ),
            (
                np.array([[0, 10]]),
                np.array([[0, 10], [1, 11]]),
                2,
                True,
                [[0], [0]],
            ),
            (
                np.array([[0, 10], [1, 11]]),
                np.array([[0, 10]]),
                2,
                True,
                [[0], [0]],
            ),
            (
                np.array([[0, 10], [1, 11], [20, 30]]),
                np.array([[0, 10], [20, 30], [20, 31]]),
                2,
                True,
                [[0, 2], [0, 1]],
            ),
        ],
    )
    def test_match_label_lists_edgecases(self, input_param, ground_truth, tolerance, one_to_one, expectation):
        output1, output2 = _match_label_lists(input_param, ground_truth, tolerance, one_to_one)

        assert_array_equal(output1, expectation[0])
        assert_array_equal(output2, expectation[1])


class TestEvaluateSegmentedStrideList:
    def _create_valid_list(self, labels):
        df = pd.DataFrame(labels, columns=["start", "end"])
        df.index.name = "s_id"
        return df

    def test_segmented_stride_list_perfect_match(self):
        list_ground_truth = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4]])
        list_predicted = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4]])
        matches = evaluate_segmented_stride_list(ground_truth=list_ground_truth, segmented_stride_list=list_predicted)

        assert np.all(matches["match_type"] == "tp")

    def test_segmented_stride_list_empty_ground_truth(self):
        list_ground_truth = self._create_valid_list([])
        list_predicted = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4]])
        matches = evaluate_segmented_stride_list(ground_truth=list_ground_truth, segmented_stride_list=list_predicted)

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

        matches = evaluate_segmented_stride_list(ground_truth=list_ground_truth, segmented_stride_list=list_predicted)

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

        matches = evaluate_segmented_stride_list(
            ground_truth=list_ground_truth, segmented_stride_list=list_predicted, tolerance=1
        )

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

        matches = evaluate_segmented_stride_list(
            ground_truth=list_ground_truth, segmented_stride_list=list_predicted, tolerance=0
        )

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

        matches = evaluate_segmented_stride_list(
            ground_truth=list_ground_truth, segmented_stride_list=list_predicted, tolerance=2, one_to_one=False
        )

        matches = _get_match_type_dfs(matches)

        assert_array_equal(matches["tp"]["s_id"].to_numpy(), [0, 1])
        assert_array_equal(matches["tp"]["s_id_ground_truth"].to_numpy(), [0, 0])

        assert matches["fp"].empty
        assert matches["fn"].empty

        assert len(list_ground_truth) != (len(matches["tp"]) + len(matches["fn"]))

    def test_segmented_stride_list_double_match_predicted_one_to_one(self):
        list_ground_truth = self._create_valid_list([[20, 30]])
        list_predicted = self._create_valid_list([[18, 30], [20, 28]])

        matches = evaluate_segmented_stride_list(
            ground_truth=list_ground_truth, segmented_stride_list=list_predicted, tolerance=2, one_to_one=True
        )

        matches = _get_match_type_dfs(matches)

        assert_array_equal(matches["tp"]["s_id"].to_numpy(), 0)
        assert_array_equal(matches["tp"]["s_id_ground_truth"].to_numpy(), [0])

        assert_array_equal(matches["fp"]["s_id"].to_numpy(), 1)
        assert_array_equal(matches["fp"]["s_id_ground_truth"].to_numpy().astype(float), np.array(np.nan))

        assert matches["fn"].empty

        assert len(list_ground_truth) == (len(matches["tp"]) + len(matches["fn"]))

    def test_segmented_multi_stride_list_perfect_match(self):
        list_ground_truth_left = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4]])
        list_predicted_left = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4]])

        list_ground_truth_right = self._create_valid_list([[3, 4], [2, 3], [1, 2], [0, 1]])
        list_predicted_right = self._create_valid_list([[3, 4], [2, 3], [1, 2], [0, 1]])

        matches = evaluate_segmented_stride_list(
            ground_truth={"left_sensor": list_ground_truth_left, "right_sensor": list_ground_truth_right},
            segmented_stride_list={"left_sensor": list_predicted_left, "right_sensor": list_predicted_right},
        )

        assert np.all(matches["left_sensor"]["match_type"] == "tp")
        assert np.all(matches["right_sensor"]["match_type"] == "tp")

    def test_segmented_multi_stride_list_empty_ground_truth(self):
        list_ground_truth_left = self._create_valid_list([])
        list_predicted_left = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4]])

        list_ground_truth_right = self._create_valid_list([])
        list_predicted_right = self._create_valid_list([[3, 4], [2, 3], [1, 2], [0, 1]])

        matches = evaluate_segmented_stride_list(
            ground_truth={"left_sensor": list_ground_truth_left, "right_sensor": list_ground_truth_right},
            segmented_stride_list={"left_sensor": list_predicted_left, "right_sensor": list_predicted_right},
        )

        matches = _get_match_type_dfs(matches)

        assert matches["left_sensor"]["tp"].empty
        assert matches["left_sensor"]["fn"].empty
        assert len(matches["left_sensor"]["fp"]) == len(list_predicted_left)
        assert_array_equal(matches["left_sensor"]["fp"]["s_id"].to_numpy(), [0, 1, 2, 3])
        assert_array_equal(
            matches["left_sensor"]["fp"]["s_id_ground_truth"].to_numpy().astype(float),
            np.array([np.nan, np.nan, np.nan, np.nan]),
        )
        assert len(list_ground_truth_left) == (len(matches["right_sensor"]["tp"]) + len(matches["right_sensor"]["fn"]))

        assert matches["right_sensor"]["tp"].empty
        assert matches["right_sensor"]["fn"].empty
        assert len(matches["right_sensor"]["fp"]) == len(list_predicted_left)
        assert_array_equal(matches["right_sensor"]["fp"]["s_id"].to_numpy(), [0, 1, 2, 3])
        assert_array_equal(
            matches["right_sensor"]["fp"]["s_id_ground_truth"].to_numpy().astype(float),
            np.array([np.nan, np.nan, np.nan, np.nan]),
        )
        assert len(list_ground_truth_right) == (len(matches["right_sensor"]["tp"]) + len(matches["right_sensor"]["fn"]))

    def test_segmented_multi_stride_list_empty_prediction(self):
        list_predicted_left = self._create_valid_list([])
        list_ground_truth_left = self._create_valid_list([[0, 1], [1, 2], [2, 3], [3, 4]])

        list_predicted_right = self._create_valid_list([])
        list_ground_truth_right = self._create_valid_list([[3, 4], [2, 3], [1, 2], [0, 1]])

        matches = evaluate_segmented_stride_list(
            ground_truth={"left_sensor": list_ground_truth_left, "right_sensor": list_ground_truth_right},
            segmented_stride_list={"left_sensor": list_predicted_left, "right_sensor": list_predicted_right},
        )

        matches = _get_match_type_dfs(matches)

        assert matches["left_sensor"]["tp"].empty
        assert matches["left_sensor"]["fp"].empty
        assert len(matches["left_sensor"]["fn"]) == len(list_ground_truth_left)
        assert_array_equal(matches["left_sensor"]["fn"]["s_id_ground_truth"].to_numpy(), [0, 1, 2, 3])
        assert_array_equal(
            matches["left_sensor"]["fn"]["s_id"].to_numpy().astype(float), [np.nan, np.nan, np.nan, np.nan]
        )
        assert len(list_ground_truth_left) == (len(matches["left_sensor"]["tp"]) + len(matches["left_sensor"]["fn"]))

        assert matches["right_sensor"]["tp"].empty
        assert matches["right_sensor"]["fp"].empty
        assert len(matches["right_sensor"]["fn"]) == len(list_ground_truth_left)
        assert_array_equal(matches["right_sensor"]["fn"]["s_id_ground_truth"].to_numpy(), [0, 1, 2, 3])
        assert_array_equal(
            matches["right_sensor"]["fn"]["s_id"].to_numpy().astype(float), [np.nan, np.nan, np.nan, np.nan]
        )
        assert len(list_ground_truth_right) == (len(matches["right_sensor"]["tp"]) + len(matches["right_sensor"]["fn"]))

    def test_segmented_multi_stride_list_match(self):
        list_ground_truth = self._create_valid_list([[20, 30], [30, 40], [40, 50], [50, 60]])
        list_predicted = self._create_valid_list([[0, 10], [11, 19], [19, 30], [30, 41], [70, 80], [80, 90]])

        matches = evaluate_segmented_stride_list(
            ground_truth={"left": list_ground_truth}, segmented_stride_list={"left": list_predicted}, tolerance=1
        )

        matches = _get_match_type_dfs(matches)

        assert_array_equal(matches["left"]["tp"]["s_id"].to_numpy(), [2, 3])
        assert_array_equal(matches["left"]["tp"]["s_id_ground_truth"].to_numpy(), [0, 1])

        assert_array_equal(matches["left"]["fp"]["s_id"].to_numpy(), [0, 1, 4, 5])
        assert_array_equal(
            matches["left"]["fp"]["s_id_ground_truth"].to_numpy().astype(float),
            np.array([np.nan, np.nan, np.nan, np.nan]),
        )

        assert_array_equal(matches["left"]["fn"]["s_id"].to_numpy().astype(float), [np.nan, np.nan])
        assert_array_equal(matches["left"]["fn"]["s_id_ground_truth"].to_numpy(), [2, 3])

        assert len(list_ground_truth) == (len(matches["left"]["tp"]) + len(matches["left"]["fn"]))

    def test_segmented_multi_stride_list_no_match(self):
        list_ground_truth = self._create_valid_list([[20, 30], [30, 40], [40, 50]])
        list_predicted = self._create_valid_list([[60, 70], [70, 80], [90, 100]])

        matches = evaluate_segmented_stride_list(
            ground_truth={"left": list_ground_truth}, segmented_stride_list={"left": list_predicted}, tolerance=0
        )

        matches = _get_match_type_dfs(matches)

        assert matches["left"]["tp"].empty

        assert_array_equal(matches["left"]["fn"]["s_id"].to_numpy().astype(float), np.array([np.nan, np.nan, np.nan]))
        assert_array_equal(matches["left"]["fn"]["s_id_ground_truth"].to_numpy(), [0, 1, 2])

        assert_array_equal(matches["left"]["fp"]["s_id"].to_numpy(), [0, 1, 2])
        assert_array_equal(
            matches["left"]["fp"]["s_id_ground_truth"].to_numpy().astype(float), np.array([np.nan, np.nan, np.nan])
        )

        assert len(list_ground_truth) == (len(matches["left"]["tp"]) + len(matches["left"]["fn"]))

    def test_segmented_multi_stride_list_double_match_predicted_many_to_one(self):
        list_ground_truth = self._create_valid_list([[20, 30]])
        list_predicted = self._create_valid_list([[18, 30], [20, 28]])

        matches = evaluate_segmented_stride_list(
            ground_truth={"left": list_ground_truth},
            segmented_stride_list={"left": list_predicted},
            tolerance=2,
            one_to_one=False,
        )

        matches = _get_match_type_dfs(matches)

        assert_array_equal(matches["left"]["tp"]["s_id"].to_numpy(), [0, 1])
        assert_array_equal(matches["left"]["tp"]["s_id_ground_truth"].to_numpy(), [0, 0])

        assert matches["left"]["fp"].empty
        assert matches["left"]["fn"].empty

        assert len(list_ground_truth) != (len(matches["left"]["tp"]) + len(matches["left"]["fn"]))

    def test_segmented_multi_stride_list_double_match_predicted_one_to_one(self):
        list_ground_truth = self._create_valid_list([[20, 30]])
        list_predicted = self._create_valid_list([[18, 30], [20, 28]])

        matches = evaluate_segmented_stride_list(
            ground_truth={"left": list_ground_truth},
            segmented_stride_list={"left": list_predicted},
            tolerance=2,
            one_to_one=True,
        )

        matches = _get_match_type_dfs(matches)

        assert_array_equal(matches["left"]["tp"]["s_id"].to_numpy(), 0)
        assert_array_equal(matches["left"]["tp"]["s_id_ground_truth"].to_numpy(), [0])

        assert_array_equal(matches["left"]["fp"]["s_id"].to_numpy(), 1)
        assert_array_equal(matches["left"]["fp"]["s_id_ground_truth"].to_numpy().astype(float), np.array(np.nan))

        assert matches["left"]["fn"].empty

        assert len(list_ground_truth) == (len(matches["left"]["tp"]) + len(matches["left"]["fn"]))

    def test_one_multi_one_single_list(self):
        multi = {"sensor": self._create_valid_list([[0, 1], [2, 3], [4, 5], [6, 7]])}
        single = self._create_valid_list([[1, 2], [3, 4], [5, 6]])

        with pytest.raises(ValidationError) as e:
            evaluate_segmented_stride_list(ground_truth=multi, segmented_stride_list=single)

        assert "not of same type" in str(e)

        with pytest.raises(ValidationError) as e:
            evaluate_segmented_stride_list(ground_truth=single, segmented_stride_list=multi)

        assert "not of same type" in str(e)
