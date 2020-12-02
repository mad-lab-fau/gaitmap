"""A helper function to evaluate the output of the temporal or spatial parameter calculation against a ground truth."""

from typing import Union, Dict, Hashable

import numpy as np
import pandas as pd

from gaitmap.utils.dataset_helper import set_correct_index
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

    By default, the output is not pretty, but this can be selected by setting `pretty_output` to `True`.
    Also by default, if a multi-senors input is given, the metrics will be calculated per sensor. If
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
        The top level has as many columns as the number of common sensors between
        the input and the ground truth. The bottom level is made out of as many columns
        as the number of parameter the respective sensor has.

    Examples
    --------
    >>> input_param = pd.DataFrame({"para1": [7, 3, 5], "para2": [7, -1, 7]}).rename_axis("stride id")
    >>> ground_truth = pd.DataFrame({"para1": [3, 6, 7], "para2": [-7, -0, 6]}).rename_axis("stride id")
    >>> print(calculate_parameter_errors(input_param, ground_truth, pretty_output=True)) #doctest: +NORMALIZE_WHITESPACE
                                          para1      para2
    mean error                        -0.333333   4.666667
    error standard deviation           3.785939   8.144528
    absolute mean error                3.000000   5.333333
    absolute error standard deviation  1.000000   7.505553
    maximal absolute error             4.000000  14.000000

    >>> pd.set_option("display.max_columns", None)
    >>> pd.set_option("display.width", 0)
    ...
    >>> input_sensor_left = pd.DataFrame(columns=["para"], data=[23, 82, 42]).rename_axis("s_id")
    >>> ground_truth_sensor_left = pd.DataFrame(columns=["para"], data=[21, 86, 65]).rename_axis("s_id")
    >>> input_sensor_right = pd.DataFrame(columns=["para"], data=[26, -58, -3]).rename_axis("s_id")
    >>> ground_truth_sensor_right = pd.DataFrame(columns=["para"], data=[96, -78, 86]).rename_axis("s_id")
    ...
    >>> print(calculate_parameter_errors(
    ...         {"left_sensor": input_sensor_left, "right_sensor": input_sensor_right},
    ...         {"left_sensor": ground_truth_sensor_left, "right_sensor": ground_truth_sensor_right}
    ... )) #doctest: +NORMALIZE_WHITESPACE
                   left_sensor right_sensor
                          para         para
    mean_error       -8.333333   -46.333333
    error_std        13.051181    58.226569
    abs_mean_error    9.666667    59.666667
    abs_error_std    11.590226    35.641736
    max_abs_error    23.000000    89.000000

    >>> print(calculate_parameter_errors(
    ...         {"left_sensor": input_sensor_left, "right_sensor": input_sensor_right},
    ...         {"left_sensor": ground_truth_sensor_left, "right_sensor": ground_truth_sensor_right},
    ...         calculate_per_sensor=False
    ... )) #doctest: +NORMALIZE_WHITESPACE
                         para
    mean_error     -27.333333
    error_std       43.098337
    abs_mean_error  34.666667
    abs_error_std   36.219700
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

    output = _calculate_error(input_parameter, ground_truth_parameter, pretty_output, calculate_per_sensor)

    if input_is_not_dict:
        output = output["__dummy__"]

    return output


def _calculate_error(
    input_parameter: Union[pd.DataFrame, Dict[Hashable, pd.DataFrame]],
    ground_truth_parameter: Union[pd.DataFrame, Dict[Hashable, pd.DataFrame], None],
    pretty_output: bool,
    calculate_per_sensor: bool = True,
) -> pd.DataFrame:
    sensor_names_list = sorted(list(set(input_parameter.keys()).intersection(ground_truth_parameter.keys())))

    if len(sensor_names_list) == 0:
        raise ValidationError("The passed parameters do not have any common sensors!")

    error_names = (
        {
            "nanmean": "mean_error",
            "nanstd": "error_std",
            "_abs_mean_error": "abs_mean_error",
            "_abs_error_std": "abs_error_std",
            "_max_abs_error": "max_abs_error",
        }
        if pretty_output is False
        else {
            "nanmean": "mean error",
            "nanstd": "error standard deviation",
            "_abs_mean_error": "absolute mean error",
            "_abs_error_std": "absolute error standard deviation",
            "_max_abs_error": "maximal absolute error",
        }
    )

    error_df = {}

    for sensor in sensor_names_list:

        try:
            input_parameter_correct = set_correct_index(input_parameter[sensor], index_cols=["s_id"])
            ground_truth_parameter_correct = set_correct_index(ground_truth_parameter[sensor], index_cols=["s_id"])
        except ValidationError:
            try:
                input_parameter_correct = set_correct_index(input_parameter[sensor], index_cols=["stride id"])
                ground_truth_parameter_correct = set_correct_index(
                    ground_truth_parameter[sensor], index_cols=["stride id"]
                )
            except ValidationError as e:
                raise ValidationError(
                    'Both inputs need to have either "s_id" or "stride id" as the index column!'
                ) from e

        common_features = sorted(
            list(set(input_parameter_correct.keys()).intersection(ground_truth_parameter_correct.keys()))
        )

        if len(common_features) == 0:
            raise ValidationError("The passed parameters do not have any common features!")

        error_df[sensor] = (
            input_parameter_correct[common_features]
            .subtract(ground_truth_parameter_correct[common_features])
            .dropna()
            .reset_index(drop=True)
        )

        if error_df[sensor].empty:
            raise ValidationError('One or more columns are empty for sensor "{}"!'.format(sensor))

    error_df = pd.concat(error_df, axis=1) if calculate_per_sensor is True else pd.concat(error_df).droplevel(0)

    # the usage of np.NamedAgg for multiple columns is still in development
    # (https://github.com/pandas-dev/pandas/pull/37627)
    # The implementation should be change to that when it is done
    return error_df.agg([np.nanmean, np.nanstd, _abs_mean_error, _abs_error_std, _max_abs_error]).rename(
        index=error_names
    )


def _abs_mean_error(x):
    return np.nanmean(np.abs(x.values))


def _abs_error_std(x):
    return np.nanstd(np.abs(x.values), ddof=1)


def _max_abs_error(x):
    return np.nanmax(np.abs(x.values))
