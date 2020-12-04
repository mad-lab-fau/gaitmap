import pandas as pd
import pytest
import numpy as np
from numpy.testing import assert_array_equal

from gaitmap.evaluation_utils import calculate_parameter_errors
from gaitmap.utils.exceptions import ValidationError


def _create_valid_input(columns, data, is_dict=False, sensors=None, mix=-1):
    if is_dict:
        output = {}
        for i, sensor_name in enumerate(sensors):
            output[sensor_name] = pd.DataFrame(columns=columns, data=data[i]).rename_axis("s_id")
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
            calculate_parameter_errors(input_param, ground_truth)

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
        output_normal = calculate_parameter_errors(input_param, ground_truth)
        output_pretty = calculate_parameter_errors(input_param, ground_truth, pretty_output=True)

        for error_type in error_types:
            assert_array_equal(np.round(output_normal["param"].loc[error_type], 5), expectation[error_type])
            assert_array_equal(
                np.round(output_pretty["param"].loc[_get_pretty_counterpart(error_type)], 5), expectation[error_type]
            )

    @pytest.mark.parametrize(
        "input_param,ground_truth,sensor_names,expectations",
        [
            (
                _create_valid_input(["param"], [np.arange(0, 10), [4, 5, 6]], is_dict=True, sensors=["1", "2"],),
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
                    {"mean_error": 0, "error_std": 0, "mean_abs_error": 0, "abs_error_std": 0, "max_abs_error": 0,},
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
        output_normal = calculate_parameter_errors(input_param, ground_truth)
        output_pretty = calculate_parameter_errors(input_param, ground_truth, pretty_output=True)

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
                {"mean_error": 0, "error_std": 0, "mean_abs_error": 0, "abs_error_std": 0, "max_abs_error": 0},
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
                },
            ),
        ],
    )
    def test_calculate_not_per_sensor_input(self, input_param, ground_truth, expectation):
        error_types = ["mean_error", "error_std", "mean_abs_error", "abs_error_std", "max_abs_error"]
        output_normal = calculate_parameter_errors(input_param, ground_truth, calculate_per_sensor=False)
        output_pretty = calculate_parameter_errors(
            input_param, ground_truth, calculate_per_sensor=False, pretty_output=True
        )

        for error_type in error_types:
            assert_array_equal(np.round(output_normal.loc[error_type], 5), expectation[error_type])
            assert_array_equal(
                np.round(output_pretty.loc[_get_pretty_counterpart(error_type)], 5), expectation[error_type]
            )
