import pandas as pd
import pytest
import numpy as np
from numpy.testing import assert_array_equal

from gaitmap.evaluation_utils import calculate_parameter_errors
from gaitmap.utils.exceptions import ValidationError


def _create_valid_input(columns, data, is_dict=False, sensor_names=None):
    if is_dict:
        output = {}
        for i, sensor_name in enumerate(sensor_names):
            output[sensor_name] = pd.DataFrame(columns=columns, data=data[i])
    else:
        output = pd.DataFrame(columns=columns, data=data)
        output.index.name = "s_id"
    return output


class TestCalculateParameterErrors:
    @pytest.mark.parametrize(
        "input_param,ground_truth,expected_error",
        [
            (
                _create_valid_input(["stride"], []),
                _create_valid_input(["stride"], []),
                'The "stride" column does not contain any data!',
            ),
            (
                _create_valid_input(["stride"], [[1]], is_dict=True, sensor_names=["1"]),
                _create_valid_input(["stride"], [[1]], is_dict=True, sensor_names=["2"]),
                "The passed parameters do not have any common sensors!",
            ),
            (
                _create_valid_input(["stride"], []),
                _create_valid_input(["stride"], [[1]], is_dict=True, sensor_names=["2"]),
                "The inputted parameters are not of same type!",
            ),
            (
                _create_valid_input(["stride"], [[1]], is_dict=True, sensor_names=["1"]),
                _create_valid_input(["not_stride"], [[1]], is_dict=True, sensor_names=["1"]),
                "The passed parameters do not have any common features!",
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
                _create_valid_input(["stride"], [1, 2, 3]),
                _create_valid_input(["stride"], [1, 2, 3]),
                {"mean_error": 0, "std_error": 0, "abs_mean_error": 0, "abs_std_error": 0, "max_abs_error": 0,},
            ),
            (
                _create_valid_input(["stride"], [7, 3, 5]),
                _create_valid_input(["stride"], [3, 6, 7]),
                {
                    "mean_error": -0.33333,
                    "std_error": 3.78594,
                    "abs_mean_error": 3.0,
                    "abs_std_error": 1.0,
                    "max_abs_error": 4.0,
                },
            ),
            (
                _create_valid_input(["stride"], [168, 265, 278.4]),
                _create_valid_input(["stride"], [99, 223, 282]),
                {
                    "mean_error": 35.8,
                    "std_error": 36.69496,
                    "abs_mean_error": 38.2,
                    "abs_std_error": 32.86518,
                    "max_abs_error": 69.0,
                },
            ),
        ],
    )
    def test_valid_input(self, input_param, ground_truth, expectation):
        error_types = ["mean_error", "std_error", "abs_mean_error", "abs_std_error", "max_abs_error"]
        output = calculate_parameter_errors(input_param, ground_truth)

        for error_type in error_types:
            assert_array_equal(np.round(output["stride"].loc[error_type], 5), expectation[error_type])

    @pytest.mark.parametrize(
        "input_param,ground_truth,expectation",
        [
            (
                _create_valid_input(["stride"], [[1, 2, 3], [4, 5, 6]], is_dict=True, sensor_names=["1", "2"]),
                _create_valid_input(["stride"], [[1, 2, 3], [4, 5, 6]], is_dict=True, sensor_names=["1", "2"]),
                {"mean_error": 0, "std_error": 0, "abs_mean_error": 0, "abs_std_error": 0, "max_abs_error": 0,},
            ),
            (
                _create_valid_input(["stride"], [[-47, 18, 7], [-32, -5, -25]], is_dict=True, sensor_names=["1", "2"]),
                _create_valid_input(["stride"], [[-9, 50, 15], [4, -38, -34]], is_dict=True, sensor_names=["1", "2"]),
                {
                    "mean_error": -12.0,
                    "std_error": 28.75413,
                    "abs_mean_error": 26.0,
                    "abs_std_error": 13.72589,
                    "max_abs_error": 38,
                },
            ),
        ],
    )
    def test_calculate_not_per_sensor_input(self, input_param, ground_truth, expectation):
        error_types = ["mean_error", "std_error", "abs_mean_error", "abs_std_error", "max_abs_error"]
        output = calculate_parameter_errors(input_param, ground_truth, calculate_per_sensor=False)

        for error_type in error_types:
            assert_array_equal(np.round(output.loc[error_type], 5), expectation[error_type])
