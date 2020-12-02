import pandas as pd
import pytest
import numpy as np
from numpy.testing import assert_array_equal

from gaitmap.evaluation_utils import calculate_parameter_errors
from gaitmap.utils.exceptions import ValidationError


def _create_valid_input(columns, data, is_dict=False, sensors=None, mix=False):
    if is_dict:
        output = {}
        for i, sensor_name in enumerate(sensors):
            output[sensor_name] = pd.DataFrame(columns=columns, data=data[i]).rename_axis("s_id")
            if mix:
                output[sensor_name] = output[sensor_name].sample(frac=1)
    else:
        output = pd.DataFrame(columns=columns, data=data).rename_axis("s_id")
    return output


class TestCalculateParameterErrors:
    @pytest.mark.parametrize(
        "input_param,ground_truth,expected_error",
        [
            (
                _create_valid_input(["param"], []),
                _create_valid_input(["param"], []),
                'One or more columns are empty for sensor "__dummy__"!',
            ),
            (
                pd.DataFrame(columns=["param"], data=[]),
                pd.DataFrame(columns=["param"], data=[]),
                'Both inputs need to have either "s_id" or "stride id" as the index column!',
            ),
            (
                _create_valid_input(["param"], [[1]], is_dict=True, sensors=["1"]),
                _create_valid_input(["param"], [[1]], is_dict=True, sensors=["2"]),
                "The passed parameters do not have any common sensors!",
            ),
            (
                _create_valid_input(["param"], []),
                _create_valid_input(["param"], [[1]], is_dict=True, sensors=["2"]),
                "The inputted parameters are not of same type!",
            ),
            (
                _create_valid_input(["param"], [[1]], is_dict=True, sensors=["1"]),
                _create_valid_input(["not_param"], [[1]], is_dict=True, sensors=["1"]),
                "The passed parameters do not have any common features!",
            ),
            (
                _create_valid_input(["param"], [[1, 2, 3, 4], []], is_dict=True, sensors=["1", "2"]),
                _create_valid_input(["param"], [[1, 2, 3, 4], [4, 6, 5]], is_dict=True, sensors=["1", "2"]),
                'One or more columns are empty for sensor "2"!',
            ),
            (
                _create_valid_input(["param"], [[1, 2, 3, 4], []], is_dict=True, sensors=["1", "2"]),
                _create_valid_input(["param"], [[1, 2, 3, 4], []], is_dict=True, sensors=["1", "2"]),
                'One or more columns are empty for sensor "2"!',
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
                {"mean_error": 0, "error_std": 0, "abs_mean_error": 0, "abs_error_std": 0, "max_abs_error": 0},
            ),
            (
                _create_valid_input(["param"], [7, 3, 5]),
                _create_valid_input(["param"], [3, 6, 7]),
                {
                    "mean_error": -0.33333,
                    "error_std": 3.78594,
                    "abs_mean_error": 3.0,
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
                    "abs_mean_error": 38.2,
                    "abs_error_std": 32.86518,
                    "max_abs_error": 69.0,
                },
            ),
        ],
    )
    def test_valid_single_sensor_input(self, input_param, ground_truth, expectation):
        error_types = ["mean_error", "error_std", "abs_mean_error", "abs_error_std", "max_abs_error"]
        output = calculate_parameter_errors(input_param, ground_truth)

        for error_type in error_types:
            assert_array_equal(np.round(output["param"].loc[error_type], 5), expectation[error_type])

    @pytest.mark.parametrize(
        "input_param,ground_truth,sensor_names,expectations",
        [
            (
                _create_valid_input(["param"], [[1, 2, 3, 4], [4, 5, 6]], is_dict=True, sensors=["1", "2"]),
                _create_valid_input(["param"], [[1, 2, 3, 4], [4, 6, 5]], is_dict=True, sensors=["1", "2"]),
                ["1", "2"],
                [
                    {"mean_error": 0, "error_std": 0, "abs_mean_error": 0, "abs_error_std": 0, "max_abs_error": 0,},
                    {
                        "mean_error": 0,
                        "error_std": 1,
                        "abs_mean_error": 0.66667,
                        "abs_error_std": 0.57735,
                        "max_abs_error": 1,
                    },
                ],
            ),
            (
                _create_valid_input(
                    ["param"], [np.arange(0, 10), [4, 5, 6]], is_dict=True, sensors=["1", "2"], mix=True
                ),
                _create_valid_input(
                    ["param"], [np.arange(0, 10), [4, 6, 5]], is_dict=True, sensors=["1", "2"], mix=True
                ),
                ["1", "2"],
                [
                    {"mean_error": 0, "error_std": 0, "abs_mean_error": 0, "abs_error_std": 0, "max_abs_error": 0,},
                    {
                        "mean_error": 0,
                        "error_std": 1,
                        "abs_mean_error": 0.66667,
                        "abs_error_std": 0.57735,
                        "max_abs_error": 1,
                    },
                ],
            ),
        ],
    )
    def test_valid_multi_sensor_input(self, input_param, ground_truth, sensor_names, expectations):
        error_types = ["mean_error", "error_std", "abs_mean_error", "abs_error_std", "max_abs_error"]
        output = calculate_parameter_errors(input_param, ground_truth)

        for sensor_name, expectation in zip(sensor_names, expectations):
            for error_type in error_types:
                assert_array_equal(np.round(output[sensor_name]["param"].loc[error_type], 5), expectation[error_type])

    @pytest.mark.parametrize(
        "input_param,ground_truth,expectation",
        [
            (
                _create_valid_input(["param"], [[1, 2, 3], [4, 5, 6]], is_dict=True, sensors=["1", "2"]),
                _create_valid_input(["param"], [[1, 2, 3], [4, 5, 6]], is_dict=True, sensors=["1", "2"]),
                {"mean_error": 0, "error_std": 0, "abs_mean_error": 0, "abs_error_std": 0, "max_abs_error": 0},
            ),
            (
                _create_valid_input(["param"], [[-47, 18, 7], [-32, -5, -25]], is_dict=True, sensors=["1", "2"]),
                _create_valid_input(["param"], [[-9, 50, 15], [4, -38, -34]], is_dict=True, sensors=["1", "2"]),
                {
                    "mean_error": -12.0,
                    "error_std": 28.75413,
                    "abs_mean_error": 26.0,
                    "abs_error_std": 13.72589,
                    "max_abs_error": 38,
                },
            ),
        ],
    )
    def test_calculate_not_per_sensor_input(self, input_param, ground_truth, expectation):
        error_types = ["mean_error", "error_std", "abs_mean_error", "abs_error_std", "max_abs_error"]
        output = calculate_parameter_errors(input_param, ground_truth, calculate_per_sensor=False)

        for error_type in error_types:
            assert_array_equal(np.round(output.loc[error_type], 5), expectation[error_type])
