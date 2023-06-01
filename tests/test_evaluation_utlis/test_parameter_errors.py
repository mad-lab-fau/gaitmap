"""

NOTE: I decided not the check every single error value and trust that the internal functions (as we are just calling
pandas functions) handle calculation of the error correctly.
"""
import doctest

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas._testing import assert_frame_equal

from gaitmap.evaluation_utils import calculate_aggregated_parameter_errors, parameter_errors
from gaitmap.evaluation_utils.parameter_errors import calculate_parameter_errors
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


class TestCalculateAggregatedParameterErrors:
    @pytest.mark.parametrize(
        ("input_param", "ground_truth", "expected_error"),
        [
            (
                _create_valid_input(["param"], []),
                _create_valid_input(["param"], []),
                "No common entries are found between predicted and reference!",
            ),
            (
                pd.DataFrame(columns=["param"], data=[]),
                pd.DataFrame(columns=["param"], data=[]),
                "Predicted and reference need to have either an index or a column named `s_id`",
            ),
            (
                _create_valid_input(["param"], [[1]], is_dict=True, sensors=["1"]),
                _create_valid_input(["param"], [[1]], is_dict=True, sensors=["2"]),
                "The predicted values and the reference do not have any common sensors",
            ),
            (
                _create_valid_input(["param"], []),
                _create_valid_input(["param"], [[1]], is_dict=True, sensors=["2"]),
                "Predicted and reference must be of the same type.",
            ),
            (
                _create_valid_input(["param"], [[1]], is_dict=True, sensors=["1"]),
                _create_valid_input(["not_param"], [[1]], is_dict=True, sensors=["1"]),
                "For sensor 1 no common parameter columns are found between predicted and reference.",
            ),
            (
                _create_valid_input(["param"], [[1, 2, 3, 4], []], is_dict=True, sensors=["1", "2"]),
                _create_valid_input(["param"], [[1, 2, 3, 4], [4, 6, 5]], is_dict=True, sensors=["1", "2"]),
                "For sensor 2 no common entries are found between predicted and reference!",
            ),
            (
                _create_valid_input(["param"], [[1, 2, 3, 4], []], is_dict=True, sensors=["1", "2"]),
                _create_valid_input(["param"], [[1, 2, 3, 4], []], is_dict=True, sensors=["1", "2"]),
                "For sensor 2 no common entries are found between predicted and reference!",
            ),
        ],
    )
    def test_invalid_input(self, input_param, ground_truth, expected_error):
        with pytest.raises(ValidationError) as e:
            calculate_aggregated_parameter_errors(predicted_parameter=input_param, reference_parameter=ground_truth)

        assert expected_error in str(e)

    @pytest.mark.parametrize(
        ("input_param", "ground_truth", "expectation"),
        [
            (
                _create_valid_input(["param"], [1, 2, 3]),
                _create_valid_input(["param"], [1, 2, 3]),
                {"error_mean": 0, "error_std": 0, "abs_error_mean": 0, "abs_error_std": 0, "abs_error_max": 0},
            ),
            (
                _create_valid_input(["param"], [7, 3, 5]),
                _create_valid_input(["param"], [3, 6, 7]),
                {
                    "error_mean": -0.33333,
                    "error_std": 3.78594,
                    "abs_error_mean": 3.0,
                    "abs_error_std": 1.0,
                    "abs_error_max": 4.0,
                },
            ),
            (
                _create_valid_input(["param"], [168, 265, 278.4]),
                _create_valid_input(["param"], [99, 223, 282]),
                {
                    "error_mean": 35.8,
                    "error_std": 36.69496,
                    "abs_error_mean": 38.2,
                    "abs_error_std": 32.86518,
                    "abs_error_max": 69.0,
                },
            ),
        ],
    )
    def test_valid_single_sensor_input(self, input_param, ground_truth, expectation):
        output_normal = calculate_aggregated_parameter_errors(
            predicted_parameter=input_param, reference_parameter=ground_truth
        )

        for error_type in expectation:
            assert_array_equal(np.round(output_normal["param"].loc[error_type], 5), expectation[error_type])

    @pytest.mark.parametrize(
        ("input_param", "ground_truth", "sensor_names", "expectations"),
        [
            (
                _create_valid_input(["param"], [np.arange(0, 10), [4, 5, 6]], is_dict=True, sensors=["1", "2"]),
                _create_valid_input(["param"], [np.arange(0, 10)[::-1], [4, 6, 5]], is_dict=True, sensors=["1", "2"]),
                ["1", "2"],
                [
                    {
                        "error_mean": 0,
                        "error_std": 6.05530,
                        "abs_error_mean": 5,
                        "abs_error_std": 2.98142,
                        "abs_error_max": 9,
                    },
                    {
                        "error_mean": 0,
                        "error_std": 1,
                        "abs_error_mean": 0.66667,
                        "abs_error_std": 0.57735,
                        "abs_error_max": 1,
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
                    {"error_mean": 0, "error_std": 0, "abs_error_mean": 0, "abs_error_std": 0, "abs_error_max": 0},
                    {
                        "error_mean": 0,
                        "error_std": 1,
                        "abs_error_mean": 0.66667,
                        "abs_error_std": 0.57735,
                        "abs_error_max": 1,
                    },
                ],
            ),
        ],
    )
    def test_valid_multi_sensor_input(self, input_param, ground_truth, sensor_names, expectations):
        output_normal = calculate_aggregated_parameter_errors(
            predicted_parameter=input_param, reference_parameter=ground_truth
        )

        for sensor_name, expectation in zip(sensor_names, expectations):
            for error_type in expectation:
                assert_array_equal(
                    np.round(output_normal[sensor_name]["param"].loc[error_type], 5), expectation[error_type]
                )

    @pytest.mark.parametrize(
        ("input_param", "ground_truth", "expectation"),
        [
            (
                _create_valid_input(["param"], [[1, 2, 3], [4, 5, 6]], is_dict=True, sensors=["1", "2"]),
                _create_valid_input(["param"], [[1, 2, 3], [4, 5, 6]], is_dict=True, sensors=["1", "2"]),
                {
                    "error_mean": 0,
                    "error_std": 0,
                    "abs_error_mean": 0,
                    "abs_error_std": 0,
                    "abs_error_max": 0,
                    "n_common": 6,
                    "n_additional_reference": 0,
                    "n_additional_predicted": 0,
                },
            ),
            (
                _create_valid_input(["param"], [[-47, 18, 7], [-32, -5, -25]], is_dict=True, sensors=["1", "2"]),
                _create_valid_input(["param"], [[-9, 50, 15], [4, -38, -34]], is_dict=True, sensors=["1", "2"]),
                {
                    "error_mean": -12.0,
                    "error_std": 28.75413,
                    "abs_error_mean": 26.0,
                    "abs_error_std": 13.72589,
                    "abs_error_max": 38,
                    "n_common": 6,
                    "n_additional_reference": 0,
                    "n_additional_predicted": 0,
                },
            ),
        ],
    )
    def test_calculate_not_per_sensor_input(self, input_param, ground_truth, expectation):
        output_normal = calculate_aggregated_parameter_errors(
            predicted_parameter=input_param, reference_parameter=ground_truth, calculate_per_sensor=False
        )

        for error_type in expectation:
            assert_array_equal(np.round(output_normal.loc[error_type], 5), expectation[error_type])

    @pytest.mark.parametrize("per_sensor", [True, False])
    def test_n_strides_missing(self, per_sensor):
        input_param = _create_valid_input(["param"], [[1, 2, 3], [4, 5, np.nan]], is_dict=True, sensors=["1", "2"])
        ground_truth = _create_valid_input(
            ["param"], [[1, np.nan, np.nan], [4, 5, 6]], is_dict=True, sensors=["1", "2"]
        )

        output = calculate_aggregated_parameter_errors(
            predicted_parameter=input_param, reference_parameter=ground_truth, calculate_per_sensor=per_sensor
        )

        if per_sensor:
            assert output["1"]["param"].loc["n_common"] == 1
            assert output["1"]["param"].loc["n_additional_reference"] == 0
            assert output["1"]["param"].loc["n_additional_predicted"] == 2
            assert output["2"]["param"].loc["n_common"] == 2
            assert output["2"]["param"].loc["n_additional_reference"] == 1
            assert output["2"]["param"].loc["n_additional_predicted"] == 0
        else:
            assert output["param"].loc["n_common"] == 3
            assert output["param"].loc["n_additional_reference"] == 1
            assert output["param"].loc["n_additional_predicted"] == 2

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

        output = calculate_aggregated_parameter_errors(
            predicted_parameter=input_param, reference_parameter=ground_truth, calculate_per_sensor=False
        )

        if single_sensor:
            param1 = output["param1"]
            param2 = output["param2"]

            assert param1.loc["n_common"] == 1
            assert param1.loc["n_additional_reference"] == 0
            assert param1.loc["n_additional_predicted"] == 2

            assert param2.loc["n_common"] == 2
            assert param2.loc["n_additional_reference"] == 1
            assert param2.loc["n_additional_predicted"] == 0
        else:
            param1 = output["param1"]
            param2 = output["param2"]

            assert param1.loc["n_common"] == 2
            assert param1.loc["n_additional_reference"] == 1
            assert param1.loc["n_additional_predicted"] == 3

            assert param2.loc["n_common"] == 4
            assert param2.loc["n_additional_reference"] == 1
            assert param2.loc["n_additional_predicted"] == 0

    def test_doctest(self):
        doctest_results = doctest.testmod(m=parameter_errors)
        assert doctest_results.failed == 0

    def test_calculate_per_sensor(self):
        input_param = _create_valid_input(["param"], [1, 2, 3], is_dict=False)
        ground_truth = _create_valid_input(["param"], [1, 2, 3], is_dict=False)
        with_per_sensor = calculate_aggregated_parameter_errors(
            predicted_parameter=input_param, reference_parameter=ground_truth, calculate_per_sensor=True
        )

        without_per_sensor = calculate_aggregated_parameter_errors(
            predicted_parameter=input_param, reference_parameter=ground_truth, calculate_per_sensor=False
        )

        assert_frame_equal(with_per_sensor, without_per_sensor)


class TestCalculateParameterErrors:
    def test_simple(self):
        predicted_parameter = _create_valid_input(
            ["param1", "param2"], [[[1, 2, 3], [4, 5, 6]]], is_dict=True, sensors=["1"]
        )
        reference_parameter = _create_valid_input(
            ["param1", "param2"], [[[1, 2, 3], [4, 5, 6]]], is_dict=True, sensors=["1"]
        )

        output = calculate_parameter_errors(
            predicted_parameter=predicted_parameter, reference_parameter=reference_parameter
        )

        assert set(output.keys()) == {"1"}
        assert set(output["1"].columns.get_level_values(1)) == {"param1", "param2"}
        assert set(output["1"].columns.get_level_values(0)) == {
            "predicted",
            "reference",
            "error",
            "abs_error",
            "rel_error",
            "abs_rel_error",
        }
