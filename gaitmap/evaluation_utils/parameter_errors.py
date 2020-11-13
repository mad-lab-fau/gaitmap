"""A helper function to evaluate output of the temporal or spatial parameter calculation against ground truth."""

from typing import Union, Dict, Hashable

import numpy as np
import pandas as pd

from gaitmap.utils.exceptions import ValidationError


def calculate_parameter_errors(
    input_parameter: Union[pd.DataFrame, Dict[Hashable, pd.DataFrame]],
    ground_truth_parameter: Union[pd.DataFrame, Dict[Hashable, pd.DataFrame]],
    pretty_output: bool = False,
    calculate_per_sensor: bool = True,
) -> pd.DataFrame:
    """Calculate 5 error metrics between a parameter input and a given ground truth.

        The metrics are: mean error, standard error, absolute mean error,
        absolute standard error, and maximal absolute error.

        By default, the output is not pretty but this can be selected by setting `pretty_output` to `True`.
        Also by default if a multi-senors input is given, the metrics will be calculatet per sensor. If
        you wish to calculate the metrics as if the data was comming from only one sensor set
        `calculate_per_sensor` to `False`.

    Parameters
    ----------
    input_parameter
        The output of the temporal or spatial parameter calculation (both `.parameters_` and `.parameters_pretty_`
        are accepted). This can be a Dataframe or a dict of such Dataframes.
    ground_truth_parameter
        The ground truth the input should be compared against.
        This must be the same type as the input is.
    pretty_output
        A bool that can be set to `True` if pretty output is preferred.
        Default is `False`.
    calculate_per_sensor
        A bool that can be set to `False` if you wish to calculate error metrics as if the
        strides were all taken by one sensor.
        Default is `True`.

    Returns
    -------
    output
        A Dataframe that has exactly 5 rows. These are the calculated error metrics
        (mean error, standard error, absolute mean error, absolute standard error,
        and maximal absolute error). The Dataframe has 2 column levels if there was
        a multi-sensor input or if not it has only 1 level.
        The 2nd level has as many columns as the number of common sensors between
        the input and the ground truth. The 1st level is made out of as many columns
        as the number of features the respective sensor has.

    Examples
    --------
    >>> input_param = pd.DataFrame({"stride1": [7, 3, 5], "stride2": [7, -1, 7]})
    >>> ground_truth = pd.DataFrame({"stride1": [3, 6, 7], "stride2": [-7, -0, 6]})
    >>> print(calculate_parameter_errors(input_param, ground_truth, pretty_output=True)) #doctest: +NORMALIZE_WHITESPACE
                              stride1    stride2
    mean error              -0.333333   4.666667
    standard error           3.785939   8.144528
    absolute mean error      3.000000   5.333333
    absolute standard error  1.000000   7.505553
    maximal absolute error   4.000000  14.000000

    >>> pd.set_option("display.max_columns", None)
    >>> pd.set_option("display.width", 0)
    ...
    >>> input_sensor_left = pd.DataFrame(columns=["stride"], data=[23, 82, 42])
    >>> ground_truth_sensor_left = pd.DataFrame(columns=["stride"], data=[21, 86, 65])
    >>> input_sensor_right = pd.DataFrame(columns=["stride"], data=[26, -58, -3])
    >>> ground_truth_sensor_right = pd.DataFrame(columns=["stride"], data=[96, -78, 86])
    ...
    >>> print(calculate_parameter_errors(
    ...         {"left_sensor": input_sensor_left, "right_sensor": input_sensor_right},
    ...         {"left_sensor": ground_truth_sensor_left, "right_sensor": ground_truth_sensor_right}
    ... )) #doctest: +NORMALIZE_WHITESPACE
                   left_sensor right_sensor
                        stride       stride
    mean_error       -8.333333   -46.333333
    std_error        13.051181    58.226569
    abs_mean_error    9.666667    59.666667
    abs_std_error    11.590226    35.641736
    max_abs_error    23.000000    89.000000

    >>> print(calculate_parameter_errors(
    ...         {"left_sensor": input_sensor_left, "right_sensor": input_sensor_right},
    ...         {"left_sensor": ground_truth_sensor_left, "right_sensor": ground_truth_sensor_right},
    ...         calculate_per_sensor=False
    ... )) #doctest: +NORMALIZE_WHITESPACE
                       stride
    mean_error     -27.333333
    std_error       43.098337
    abs_mean_error  34.666667
    abs_std_error   36.219700
    max_abs_error   89.000000

    See Also
    --------
    gaitmap.parameters.TemporalParameterCalculation
    gaitmap.parameters.SpatialParameterCalculation

    """
    input_is_not_dict = not isinstance(input_parameter, dict)
    ground_truth_is_not_dict = not isinstance(ground_truth_parameter, dict)

    if input_is_not_dict != ground_truth_is_not_dict:
        raise ValidationError("The inputted parameters are not of same type!")

    if input_is_not_dict:
        input_parameter = {"__dummy__": input_parameter}
        ground_truth_parameter = {"__dummy__": ground_truth_parameter}

    sensor_names_list = sorted(list(set(input_parameter.keys()).intersection(ground_truth_parameter.keys())))

    if not sensor_names_list:
        raise ValidationError("The passed parameters do not have any common sensors!")

    if not calculate_per_sensor:
        input_parameter = {
            "__calculate_not_per_sensor__": pd.concat(
                [input_parameter[sensor_name] for sensor_name in sensor_names_list]
            )
        }

        ground_truth_parameter = {
            "__calculate_not_per_sensor__": pd.concat(
                [ground_truth_parameter[sensor_name] for sensor_name in sensor_names_list]
            )
        }

        sensor_names_list = ["__calculate_not_per_sensor__"]

    sensor_df = {}

    for sensor_name in sensor_names_list:
        sensor_df[sensor_name] = _calculate_error(
            input_parameter[sensor_name], ground_truth_parameter[sensor_name], pretty_output
        )

    output = pd.concat(sensor_df, axis=1)

    if input_is_not_dict:
        output = output["__dummy__"]

    if not calculate_per_sensor:
        output = output["__calculate_not_per_sensor__"]

    return output


def _calculate_error(
    input_parameter: Union[pd.DataFrame, Dict[Hashable, pd.DataFrame]],
    ground_truth_parameter: Union[pd.DataFrame, Dict[Hashable, pd.DataFrame]],
    pretty: bool,
) -> pd.DataFrame:
    # Absolutely necessary to do this like that because when the aggregate function is used it
    # sets the indices to "mean", "std" and "amax". This makes it impossible to do the renaming in one
    # go so I had to create 2 different dicts for renaming.
    error_names_1 = (
        {"mean": "mean_error", "std": "std_error"} if not pretty else {"mean": "mean error", "std": "standard error"}
    )

    error_names_2 = (
        {"mean": "abs_mean_error", "std": "abs_std_error", "amax": "max_abs_error"}
        if not pretty
        else {"mean": "absolute mean error", "std": "absolute standard error", "amax": "maximal absolute error"}
    )

    common_features = sorted(list(set(input_parameter.keys()).intersection(ground_truth_parameter.keys())))
    if not common_features:
        raise ValidationError("The passed parameters do not have any common features!")

    error_dfs = []

    for feature in common_features:
        if len(input_parameter[feature]) == 0:
            raise ValidationError('The "{}" column does not contain any data!'.format(feature))

        error = input_parameter[feature].sort_index() - ground_truth_parameter[feature].sort_index()

        error_df = pd.DataFrame(error).dropna().reset_index(drop=True)
        error_df[feature + "_abs"] = np.abs(error_df[feature])
        error_df = error_df.aggregate([np.mean, np.std, np.max], axis=0)
        error_df = (
            error_df[feature]
            .drop(index="amax")
            .rename(index=error_names_1)
            .append(error_df[feature + "_abs"].rename(index=error_names_2))
        )
        error_df = pd.DataFrame(error_df, columns=[feature])

        if len(error) == 0:
            raise ValidationError('No "s_id"s match for the "{}" column!'.format(feature))

        error_dfs.append(error_df)

    return pd.concat(error_dfs, axis=1)
