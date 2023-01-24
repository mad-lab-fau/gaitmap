import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from gaitmap.evaluation_utils import calculate_parameter_errors
from gaitmap.utils.exceptions import ValidationError


def _create_valid_input(columns, data, is_dict=False, sensors=None, mix=-1):
    if is_dict:
        output = {}
        for i, sensor_name in enumerate(sensors):
            if len(columns) > 1:
                output[sensor_name] = pd.DataFrame(dict(zip(columns, data[i]))).rename_axis("s_id")
            else:
                output[sensor_name] = pd.DataFrame(data[i], columns=columns).rename_axis("s_id")
            if mix > -1:
                output[sensor_name] = output[sensor_name].sample(frac=1, random_state=mix)
    else:
        output = pd.DataFrame(columns=columns, data=data).rename_axis("s_id")
    return output


def _get_pretty_counterpart(normal_error):
    temp = {
        "mean_error": "mean error",
        "error_std": "error standard deviation",
        "mean_abs_error": "mean absolute error",
        "abs_error_std": "absolute error standard deviation",
        "max_abs_error": "maximal absolute error",
        "n_common": "number common entries",
        "n_additional_ground_truth": "number additional entries ground truth",
        "n_additional_input": "number of additional entries input",
    }

    return temp[normal_error]


class TestCalculateParameterErrors:
    @pytest.mark.parametrize(
        "input_param,ground_truth,expected_error",
        [
            (
                _create_valid_input(["param"], []),
                _create_valid_input(["param"], []),
                "No common strides are found between input and reference!",
            ),
            (
                pd.DataFrame(columns=["param"], data=[]),
                pd.DataFrame(columns=["param"], data=[]),
                'Inputs and reference need to have either an index or a column named "s_id" or "stride id"',
            ),
            (
                _create_valid_input(["param"], [[1]], is_dict=True, sensors=["1"]),
                _create_valid_input(["param"], [[1]], is_dict=True, sensors=["2"]),
                "The input and reference do not have any common sensors",
            ),
            (
                _create_valid_input(["param"], []),
                _create_valid_input(["param"], [[1]], is_dict=True, sensors=["2"]),
                "Input and reference must be of the same type.",
            ),
            (
                _create_valid_input(["param"], [[1]], is_dict=True, sensors=["1"]),
                _create_valid_input(["not_param"], [[1]], is_dict=True, sensors=["1"]),
                "For sensor 1 no common parameter columns are found between input and reference.",
            ),
            (
                _create_valid_input(["param"], [[1, 2, 3, 4], []], is_dict=True, sensors=["1", "2"]),
                _create_valid_input(["param"], [[1, 2, 3, 4], [4, 6, 5]], is_dict=True, sensors=["1", "2"]),
                "For sensor 2 no common strides are found between input and reference!",
            ),
            (
                _create_valid_input(["param"], [[1, 2, 3, 4], []], is_dict=True, sensors=["1", "2"]),
                _create_valid_input(["param"], [[1, 2, 3, 4], []], is_dict=True, sensors=["1", "2"]),
                "For sensor 2 no common strides are found between input and reference!",
            ),
        ],
    )
    def test_invalid_input(self, input_param, ground_truth, expected_error):
        with pytest.raises(ValidationError) as e:
            calculate_parameter_errors(input_parameter=input_param, ground_truth_parameter=ground_truth)

        assert expected_error in str(e)

    @pytest.mark.parametrize(
        "input_param,ground_truth,expectation",
        [
            (
                _create_valid_input(["param"], [1, 2, 3]),
                _create_valid_input(["param"], [1, 2, 3]),
                {"mean_error": 0, "error_std": 0, "mean_abs_error": 0, "abs_error_std": 0, "max_abs_error": 0},
            ),
            (
                _create_valid_input(["param"], [7, 3, 5]),
                _create_valid_input(["param"], [3, 6, 7]),
                {
                    "mean_error": -0.33333,
                    "error_std": 3.78594,
                    "mean_abs_error": 3.0,
                    "abs_error_std": 1.0,
                    "max_abs_error": 4.0,
                },
            ),
            (
                _create_valid_input(["param"], [168, 265, 278.4]),
                _create_valid_input(["param"], [99, 223, 282]),
                {
                    "mean_error": 35.8,
                    "error_std": 36.69496,
                    "mean_abs_error": 38.2,
                    "abs_error_std": 32.86518,
                    "max_abs_error": 69.0,
                },
            ),
        ],
    )
    def test_valid_single_sensor_input(self, input_param, ground_truth, expectation):
        error_types = ["mean_error", "error_std", "mean_abs_error", "abs_error_std", "max_abs_error"]
        output_normal = calculate_parameter_errors(input_parameter=input_param, ground_truth_parameter=ground_truth)
        output_pretty = calculate_parameter_errors(
            input_parameter=input_param, ground_truth_parameter=ground_truth, pretty_output=True
        )

        for error_type in error_types:
            assert_array_equal(np.round(output_normal["param"].loc[error_type], 5), expectation[error_type])
            assert_array_equal(
                np.round(output_pretty["param"].loc[_get_pretty_counterpart(error_type)], 5), expectation[error_type]
            )

    @pytest.mark.parametrize(
        "input_param,ground_truth,sensor_names,expectations",
        [
            (
                _create_valid_input(["param"], [np.arange(0, 10), [4, 5, 6]], is_dict=True, sensors=["1", "2"]),
                _create_valid_input(["param"], [np.arange(0, 10)[::-1], [4, 6, 5]], is_dict=True, sensors=["1", "2"]),
                ["1", "2"],
                [
                    {
                        "mean_error": 0,
                        "error_std": 6.05530,
                        "mean_abs_error": 5,
                        "abs_error_std": 2.98142,
                        "max_abs_error": 9,
                    },
                    {
                        "mean_error": 0,
                        "error_std": 1,
                        "mean_abs_error": 0.66667,
                        "abs_error_std": 0.57735,
                        "max_abs_error": 1,
                    },
                ],
            ),
            (
                _create_valid_input(
                    ["param"], [np.arange(0, 10), [4, 5, 6]], is_dict=True, sensors=["1", "2"], mix=9149
                ),
                _create_valid_input(
                    ["param"], [np.arange(0, 10), [4, 6, 5]], is_dict=True, sensors=["1", "2"], mix=1516
                ),
                ["1", "2"],
                [
                    {"mean_error": 0, "error_std": 0, "mean_abs_error": 0, "abs_error_std": 0, "max_abs_error": 0},
                    {
                        "mean_error": 0,
                        "error_std": 1,
                        "mean_abs_error": 0.66667,
                        "abs_error_std": 0.57735,
                        "max_abs_error": 1,
                    },
                ],
            ),
        ],
    )
    def test_valid_multi_sensor_input(self, input_param, ground_truth, sensor_names, expectations):
        error_types = ["mean_error", "error_std", "mean_abs_error", "abs_error_std", "max_abs_error"]
        output_normal = calculate_parameter_errors(input_parameter=input_param, ground_truth_parameter=ground_truth)
        output_pretty = calculate_parameter_errors(
            input_parameter=input_param, ground_truth_parameter=ground_truth, pretty_output=True
        )

        for sensor_name, expectation in zip(sensor_names, expectations):
            for error_type in error_types:
                assert_array_equal(
                    np.round(output_normal[sensor_name]["param"].loc[error_type], 5), expectation[error_type]
                )
                assert_array_equal(
                    np.round(output_pretty[sensor_name]["param"].loc[_get_pretty_counterpart(error_type)], 5),
                    expectation[error_type],
                )

    @pytest.mark.parametrize(
        "input_param,ground_truth,expectation",
        [
            (
                _create_valid_input(["param"], [[1, 2, 3], [4, 5, 6]], is_dict=True, sensors=["1", "2"]),
                _create_valid_input(["param"], [[1, 2, 3], [4, 5, 6]], is_dict=True, sensors=["1", "2"]),
                {
                    "mean_error": 0,
                    "error_std": 0,
                    "mean_abs_error": 0,
                    "abs_error_std": 0,
                    "max_abs_error": 0,
                    "n_common_strides": 6,
                    "n_additional_ground_truth": 0,
                    "n_additional_input": 0,
                },
            ),
            (
                _create_valid_input(["param"], [[-47, 18, 7], [-32, -5, -25]], is_dict=True, sensors=["1", "2"]),
                _create_valid_input(["param"], [[-9, 50, 15], [4, -38, -34]], is_dict=True, sensors=["1", "2"]),
                {
                    "mean_error": -12.0,
                    "error_std": 28.75413,
                    "mean_abs_error": 26.0,
                    "abs_error_std": 13.72589,
                    "max_abs_error": 38,
                    "n_common_strides": 6,
                    "n_additional_ground_truth": 0,
                    "n_additional_input": 0,
                },
            ),
        ],
    )
    def test_calculate_not_per_sensor_input(self, input_param, ground_truth, expectation):
        output_normal = calculate_parameter_errors(
            input_parameter=input_param, ground_truth_parameter=ground_truth, calculate_per_sensor=False
        )
        output_pretty = calculate_parameter_errors(
            input_parameter=input_param,
            ground_truth_parameter=ground_truth,
            pretty_output=True,
            calculate_per_sensor=False,
        )

        for error_type in expectation.keys():
            assert_array_equal(np.round(output_normal.loc[error_type], 5), expectation[error_type])
            assert_array_equal(
                np.round(output_pretty.loc[_get_pretty_counterpart(error_type)], 5), expectation[error_type]
            )

    @pytest.mark.parametrize("per_sensor", [True, False])
    def test_n_strides_missing(self, per_sensor):
        input_param = _create_valid_input(["param"], [[1, 2, 3], [4, 5, np.nan]], is_dict=True, sensors=["1", "2"])
        ground_truth = _create_valid_input(
            ["param"], [[1, np.nan, np.nan], [4, 5, 6]], is_dict=True, sensors=["1", "2"]
        )

        output = calculate_parameter_errors(
            input_parameter=input_param, ground_truth_parameter=ground_truth, calculate_per_sensor=per_sensor
        )

        if per_sensor:
            assert output["1"]["param"].loc["n_common"] == 1
            assert output["1"]["param"].loc["n_additional_ground_truth"] == 0
            assert output["1"]["param"].loc["n_additional_input"] == 2
            assert output["2"]["param"].loc["n_common"] == 2
            assert output["2"]["param"].loc["n_additional_ground_truth"] == 1
            assert output["2"]["param"].loc["n_additional_input"] == 0
        else:
            assert output["param"].loc["n_common"] == 3
            assert output["param"].loc["n_additional_ground_truth"] == 1
            assert output["param"].loc["n_additional_input"] == 2

    @pytest.mark.parametrize("single_sensor", [True, False])
    def test_n_strides_missing_multi_param(self, single_sensor):
        if single_sensor:
            input_param = _create_valid_input(
                ["param1", "param2"], [[[1, 2, 3], [4, 5, np.nan]]], is_dict=True, sensors=["1"]
            )
            ground_truth = _create_valid_input(
                ["param1", "param2"], [[[1, np.nan, np.nan], [4, 5, 6]]], is_dict=True, sensors=["1"]
            )
        else:
            input_param = _create_valid_input(
                ["param1", "param2"],
                [[[1, 2, 3], [4, 5, np.nan]], [[7, 8, np.nan], [10, 11, np.nan]]],
                is_dict=True,
                sensors=["1", "2"],
            )
            ground_truth = _create_valid_input(
                ["param1", "param2"],
                [[[1, np.nan, np.nan], [4, 5, 6]], [[7, np.nan, 9], [10, 11, np.nan]]],
                is_dict=True,
                sensors=["1", "2"],
            )

        output = calculate_parameter_errors(
            input_parameter=input_param, ground_truth_parameter=ground_truth, calculate_per_sensor=False
        )

        if single_sensor:
            param1 = output["param1"]
            param2 = output["param2"]

            assert param1.loc["n_common"] == 1
            assert param1.loc["n_additional_ground_truth"] == 0
            assert param1.loc["n_additional_input"] == 2

            assert param2.loc["n_common"] == 2
            assert param2.loc["n_additional_ground_truth"] == 1
            assert param2.loc["n_additional_input"] == 0
        else:
            param1 = output["param1"]
            param2 = output["param2"]

            assert param1.loc["n_common"] == 2
            assert param1.loc["n_additional_ground_truth"] == 1
            assert param1.loc["n_additional_input"] == 3

            assert param2.loc["n_common"] == 4
            assert param2.loc["n_additional_ground_truth"] == 1
            assert param2.loc["n_additional_input"] == 0
