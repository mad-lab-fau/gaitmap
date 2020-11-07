"""A helper function to evaluate output of the temporal or spatial parameter calculation against ground truth."""

from typing import Union, Dict, Hashable

import pandas as pd
import numpy as np
from scipy.stats import sem
from sklearn.metrics import mean_absolute_error
from gaitmap.utils.exceptions import ValidationError


def calculate_parameter_errors(
    input_parameter: Union[pd.DataFrame, Dict[Hashable, pd.DataFrame]],
    ground_truth_parameter: Union[pd.DataFrame, Dict[Hashable, pd.DataFrame]],
    pretty_output: bool = False,
) -> pd.DataFrame:
    """Calculate 5 error metrics between a parameter input and a given ground truth.

        The metrics are: error, absolute error, standard mean error, mean absolute error,
        and the maximal error.

        By default, the output is not pretty but this can be selected by setting `pretty_output` to `True`.

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

    Returns
    -------
    output
        A Dataframe that has 2 column levels per sensor if there was a multi-sensor input
        or if not it has only 1 level. The 2nd level has as many columns as the number of common sensors between the
        input and the ground truth. The 1st level is always made out of 5 columns. These are
        the calculated error metrics (error, absolute error, standard mean error, mean absolute error,
        and the maximal error). The 0th column level contains the calculated errors. The error and absolute error
        will use all rows for output whereas the standard mean error, mean absolute error,
        and the maximal error only use the 0st row after which every row will be `np.NaN`.
        For clarification see examples below.

    Examples
    --------
    >>> input = pd.DataFrame(columns=["stride"], data=[7, 3, 5])
    >>> ground_truth = pd.DataFrame(columns=["stride"], data=[3, 6, 7])
    >>> print(calculate_parameter_errors(input, ground_truth, pretty_output=True)) #doctest: +NORMALIZE_WHITESPACE
          error absolute error standard error absolute standard error maximal error
         stride         stride         stride                  stride        stride
    s_id
    0         4              4        0.57735                     3.0           4.0
    1        -3              3            NaN                     NaN           NaN
    2        -2              2            NaN                     NaN           NaN

    >>> pd.set_option("display.max_columns", None)
    >>> pd.set_option("display.width", 0)
    ...
    >>> input_sensor = pd.DataFrame(columns=["stride"], data=[23, 82, 42])
    >>> ground_truth_sensor = pd.DataFrame(columns=["stride"], data=[21, 86, 65])
    ...
    >>> print(calculate_parameter_errors(
    ...         {"test_sensor": input_sensor},
    ...         {"test_sensor": ground_truth_sensor}
    ... )) #doctest: +NORMALIZE_WHITESPACE
          test_sensor
               error abs_error std_error abs_std_error max_error
              stride    stride    stride        stride    stride
    s_id
    0              2         2   6.69162      9.666667      23.0
    1             -4         4       NaN           NaN       NaN
    2            -23        23       NaN           NaN       NaN

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

    sensor_df = {}

    for sensor_name in sensor_names_list:
        sensor_df[sensor_name] = _calculate_error(
            input_parameter[sensor_name], ground_truth_parameter[sensor_name], pretty_output
        )

    output = pd.concat(sensor_df, axis=1)
    output.index.name = "s_id"

    if input_is_not_dict:
        output = output["__dummy__"]

    return output


def _calculate_error(
    input_parameter: Union[pd.DataFrame, Dict[Hashable, pd.DataFrame]],
    ground_truth_parameter: Union[pd.DataFrame, Dict[Hashable, pd.DataFrame]],
    pretty: bool,
) -> pd.DataFrame:
    error_names = (
        ["error", "abs_error", "std_error", "abs_std_error", "max_error"]
        if not pretty
        else ["error", "absolute error", "standard error", "absolute standard error", "maximal error"]
    )

    error_dicts = [{} for _ in range(len(error_names))]

    for key in input_parameter.keys():

        if len(input_parameter[key]) == 0:
            raise ValidationError('The "{}" column does not contain any data!'.format(key))

        error = input_parameter[key] - ground_truth_parameter[key]
        abs_error = np.abs(error)

        error_setter = {
            0: error,
            1: abs_error,
            2: [sem(abs_error)],
            3: [mean_absolute_error(ground_truth_parameter[key], input_parameter[key])],
            4: [np.max(abs_error)],
        }

        for i in range(len(error_names)):
            error_dicts[i][key] = error_setter.get(i)

    error_dfs = [_create_sub_df(error_dicts[i], error_names[i]) for i in range(len(error_names))]

    return pd.concat(error_dfs, axis=1)


def _create_sub_df(input_dict, name_of_error):
    output = pd.DataFrame(input_dict).reset_index(drop=True)
    output.columns = pd.MultiIndex.from_product([[name_of_error], output.columns])

    return output
