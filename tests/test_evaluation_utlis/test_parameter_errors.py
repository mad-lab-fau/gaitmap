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
            output[sensor_name] = pd.DataFrame(columns=columns[i], data=data[i])
    else:
        output = pd.DataFrame(columns=columns, data=data)
        output.index.name = "s_id"
    return output


class TestCalculateParameterErrors:
    @pytest.mark.parametrize(
        "input,ground_truth,expected_error",
        [
            (
                _create_valid_input(["stride"], []),
                _create_valid_input(["stride"], []),
                'The "stride" column does not contain any data!',
            ),
            (
                _create_valid_input([["stride"]], [[1]], is_dict=True, sensor_names=["1"]),
                _create_valid_input([["stride"]], [[1]], is_dict=True, sensor_names=["2"]),
                "The passed parameters do not have any common sensors!",
            ),
            (
                _create_valid_input(["stride"], []),
                _create_valid_input([["stride"]], [[1]], is_dict=True, sensor_names=["2"]),
                "The inputted parameters are not of same type!",
            ),
        ],
    )
    def test_invalid_input(self, input, ground_truth, expected_error):
        with pytest.raises(ValidationError) as e:
            calculate_parameter_errors(input, ground_truth)

        assert expected_error in str(e)

    @pytest.mark.parametrize(
        "input,ground_truth,expectation",
        [
            (
                _create_valid_input(["stride"], [1, 2, 3]),
                _create_valid_input(["stride"], [1, 2, 3]),
                {
                    "error": [0, 0, 0],
                    "abs_error": [0, 0, 0],
                    "std_error": [0, np.nan, np.nan],
                    "abs_std_error": [0, np.nan, np.nan],
                    "max_error": [0, np.nan, np.nan],
                },
            ),
            (
                _create_valid_input(["stride"], [7, 3, 5]),
                _create_valid_input(["stride"], [3, 6, 7]),
                {
                    "error": [4, -3, -2],
                    "abs_error": [4, 3, 2],
                    "std_error": [0.57735, np.nan, np.nan],
                    "abs_std_error": [3.0, np.nan, np.nan],
                    "max_error": [4, np.nan, np.nan],
                },
            ),
            (
                _create_valid_input(["stride"], [168, 265, 278.4]),
                _create_valid_input(["stride"], [99, 223, 282]),
                {
                    "error": [69, 42.0, -3.60],
                    "abs_error": [69, 42.0, 3.60],
                    "std_error": [18.97472, np.nan, np.nan],
                    "abs_std_error": [38.2, np.nan, np.nan],
                    "max_error": [69, np.nan, np.nan],
                },
            ),
        ],
    )
    def test_valid_input(self, input, ground_truth, expectation):
        error_types = ["error", "abs_error", "std_error", "abs_std_error", "max_error"]
        output = calculate_parameter_errors(input, ground_truth)

        for error_type in error_types:
            assert_array_equal(
                np.round(output[error_type]["stride"].to_numpy().astype(np.float), 5), expectation[error_type]
            )
