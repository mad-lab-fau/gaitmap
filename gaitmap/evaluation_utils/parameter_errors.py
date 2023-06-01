"""A helper function to evaluate the output of the temporal or spatial parameter calculation against a ground truth."""
import warnings
from typing import Dict, Literal, Tuple, Union

import numpy as np
import pandas as pd

from gaitmap.utils._types import _Hashable
from gaitmap.utils.datatype_helper import set_correct_index
from gaitmap.utils.exceptions import ValidationError

_ID_COL_NAME = "__id_col__"


def calculate_parameter_errors(
    *,
    reference_parameter: Union[pd.DataFrame, Dict[_Hashable, pd.DataFrame]],
    predicted_parameter: Union[pd.DataFrame, Dict[_Hashable, pd.DataFrame]],
    id_column: str = "s_id",
) -> Tuple[Union[pd.DataFrame, Dict[_Hashable, pd.DataFrame]], Union[pd.DataFrame, Dict[_Hashable, pd.DataFrame]]]:
    """Calculate the error per row between a parameter predicted and a given ground truth.

    We calculate four different groups of errors:

        - The error between the predicted and the reference value (`predicted - reference`)
        - The relative error between the predicted and the reference value (`(predicted - reference) / reference`)
        - The absolute error between the predicted and the reference value (`abs(predicted - reference)`)
        - The absolute relative error between the predicted and the reference value (`abs(predicted - reference) /
          abs(reference)`)

    The output dataframe also contains the reference and predicted values, so that all information is available in one
    place.

    All errors are calculated by first aligning the reference and predicted values based on the `id_column` (i.e. we
    drop all values that are not present in both dataframes).
    The same happens for the columns (i.e. parameters) of the dataframes.
    Errors are only calculated for parameters that are present in both dataframes.

    If the input is a multi-sensor parameter list (i.e. a dict of dataframes) the alignment and error calculation is
    done for each sensor and the output has the same structure as the input.

    Parameters
    ----------
    reference_parameter
        The reference the predicted values should be compared against.
        This must be the same type (i.e. single/multi sensor) as the predicted input.
        Further, sensor names, column names, and unique ids must match with the `predicted_parameters`.
    predicted_parameter
        The predicted parameter values.
        Usually, this is the output of the temporal or spatial parameter calculation.
        But, you can also pass a custom calculation/aggregation.
        Make sure you adjust the `id_column` parameter accordingly.
        This can be a Dataframe or a dict of such Dataframes.
    id_column
        The name of the column/index that contains unique entry ids.
        This will be used to align the predicted and reference parameters.
        For a normal output of the temporal or spatial parameter calculation, use `id_column="s_id"` (default) column.
        If you are using the "pretty" output of these calculations, use `id_column="stride id"`.
        For custom calculations/aggregations, make sure you have a column/index that contains unique ids.

    Returns
    -------
    error_df
        A Dataframe/Dict of Dataframes with a mult-columns.
        The first level represents the parameter name, the second level the error type (see above).
    common_rows_stats
        A Dataframe/Dict of Dataframes representing the statistics of the alignment.
        I.e. how many datapoints were present in both the reference and the predicted parameters.
        This can be different per parameter.Parameters

    """
    # TODO: Add proper parameter validation (see issue #150)
    predicted_is_not_dict = not isinstance(predicted_parameter, dict)
    reference_is_not_dict = not isinstance(reference_parameter, dict)

    if predicted_is_not_dict != reference_is_not_dict:
        raise ValidationError(
            "Predicted and reference must be of the same type. "
            "Both need to be either single or multi-sensor parameter lists."
        )

    if predicted_is_not_dict:
        predicted_parameter = {"__dummy__": predicted_parameter}
        reference_parameter = {"__dummy__": reference_parameter}

    aligned_parameters, common_rows = _align_parameters(reference_parameter, predicted_parameter, id_column=id_column)
    output = _calculate_error(aligned_parameters)

    if predicted_is_not_dict:
        output = output["__dummy__"]
        common_rows = common_rows["__dummy__"]

    return output, common_rows


def calculate_aggregated_parameter_errors(
    *,
    reference_parameter: Union[pd.DataFrame, Dict[_Hashable, pd.DataFrame]],
    predicted_parameter: Union[pd.DataFrame, Dict[_Hashable, pd.DataFrame]],
    calculate_per_sensor: bool = True,
    scoring_errors: Literal["ignore", "warn", "raise"] = "warn",
    id_column: str = "s_id",
) -> pd.DataFrame:
    """Calculate various error metrics between a parameter predicted and a given ground truth.

    This method can be applied to stride level parameters or aggregated parameters over a gait test/participant/... .
    In both cases the reference and the predicted values are simply aligned based on the specified `id_column`.
    All non-common entries are ignored for the calculation of the error metrics.

    In general, we calculate four different groups of errors:

    - The error between the predicted and the reference value (`predicted - reference`)
    - The relative error between the predicted and the reference value (`(predicted - reference) / reference`)
    - The absolute error between the predicted and the reference value (`abs(predicted - reference)`)
    - The absolute relative error between the predicted and the reference value (`abs(predicted - reference) /
      abs(reference)`)

    For each of these groups of errors, we calculate the maximum, minimum, mean, median, standard deviation, the
    0.05/0.95 quantiles, and the upper/lower limit of aggreement (loa).
    In addition the ICC (intraclass correlation coefficient) with the respective 0.05/0.95 quantiles is calculated.
    All metrics are calculated for all columns that are available in both, the predicted parameters and the reference.
    It is up to you, to decide if a specific error metric makes sense for a given parameter.

    In addition, the number of common entries (based on the `id_column`), additional entries in the reference and
    additional entries in the predicted values are calculated.
    These metrics are helpful, as parameter errors are only calculated for entries that are present in both the
    inputs.
    Entries between the predicted and the reference are matched based on the column/index name specified by the
    `id_column` paramter.
    For a normal output of the temporal or spatial parameter calculation, use `id_column="s_id"` column.
    If you are using the "pretty" output of these calculations, use `id_column="stride id"`.
    In case you used custom calculations/aggregations, make sure you have a column/index that contains unique ids that
    match correctly between the predicted and the reference parameters.
    Parameters with `np.nan` are considered to be missing for the respective entry.

    The metrics can either be calculated per sensor or for all sensors combined (see the `calculate_per_sensor`
    parameter).
    In the latter case, the error per entry is calculated and then all entry of all sensors combined before
    calculating the summary metrics (mean, std, ...).
    This might be desired, if you have one sensor per foot, but want to have statistics over all entries independent of
    the foot.

    Parameters
    ----------
    reference_parameter
        The reference the predicted values should be compared against.
        This must be the same type (i.e. single/multi sensor) as the predicted input.
        Further, sensor names, column names, and unique ids must match with the `predicted_parameters`.
    predicted_parameter
        The predicted parameter values.
        Usually, this is the output of the temporal or spatial parameter calculation.
        But, you can also pass a custom calculation/aggregation.
        Make sure you adjust the `id_column` parameter accordingly.
        This can be a Dataframe or a dict of such Dataframes.
    calculate_per_sensor
        A bool that can be set to `False` if you wish to calculate error metrics as if the entries were all taken by
        one sensor.
        Default is `True`.
    scoring_errors
        How to handle errors during the scoring.
        Can be one of `ignore`, `warn`, or `raise`.
        Default is `warn`.
        At the moment, this only effects the calculation of the ICC.
        In case of ignore, we will also ignore warnings that might be raised during the calculation.
        In all cases the value for a given metric is set to `np.nan`.
    id_column
        The name of the column/index that contains unique entry ids.
        This will be used to align the predicted and reference parameters.
        For a normal output of the temporal or spatial parameter calculation, use `id_column="s_id"` (default) column.
        If you are using the "pretty" output of these calculations, use `id_column="stride id"`.
        For custom calculations/aggregations, make sure you have a column/index that contains unique ids.

    Returns
    -------
    output
        A Dataframe with one row per error metric and one column per parameter.
        In case of a multi-sensor predicted (and `calculate_per_sensor=True`), the dataframe has 2 column levels.
        The first level is the sensor name and the second one the parameter name.

    Examples
    --------
    >>> predicted_param = pd.DataFrame({"para1": [7, 3, 5, 9], "para2": [7, -1, 7, -6]}).rename_axis("trial id")
    >>> reference = pd.DataFrame({"para1": [3, 6, 7, 8], "para2": [-7, -1, 6, -5]}).rename_axis("trial id")
    >>> calculate_aggregated_parameter_errors(
    ...     predicted_parameter=predicted_param,
    ...     reference_parameter=reference,
    ...     id_column="trial id",
    ... )  # doctest: +NORMALIZE_WHITESPACE
                                 para1      para2
    predicted_mean            6.000000   1.750000
    reference_mean            6.000000  -1.750000
    error_mean                0.000000   3.500000
    abs_error_mean            2.500000   4.000000
    rel_error_mean            0.168155  -0.408333
    abs_rel_error_mean        0.561012   0.591667
    predicted_std             2.581989   6.396614
    reference_std             2.160247   5.737305
    error_std                 3.162278   7.047458
    abs_error_std             1.290994   6.683313
    rel_error_std             0.818928   1.064712
    abs_rel_error_std         0.537307   0.942956
    predicted_median          6.000000   3.000000
    reference_median          6.500000  -3.000000
    error_median             -0.500000   0.500000
    abs_error_median          2.500000   1.000000
    rel_error_median         -0.080357   0.083333
    abs_rel_error_median      0.392857   0.183333
    predicted_q05             3.300000  -5.250000
    reference_q05             3.450000  -6.700000
    error_q05                -2.850000  -0.850000
    abs_error_q05             1.150000   0.150000
    rel_error_q05            -0.467857  -1.700000
    abs_rel_error_q05         0.149107   0.025000
    predicted_q95             8.700000   7.000000
    reference_q95             7.850000   4.950000
    error_q95                 3.550000  12.050000
    abs_error_q95             3.850000  12.050000
    rel_error_q95             1.152083   0.195000
    abs_rel_error_q95         1.208333   1.730000
    predicted_max             9.000000   7.000000
    reference_max             8.000000   6.000000
    error_max                 4.000000  14.000000
    abs_error_max             4.000000  14.000000
    rel_error_max             1.333333   0.200000
    abs_rel_error_max         1.333333   2.000000
    predicted_min             3.000000  -6.000000
    reference_min             3.000000  -7.000000
    error_min                -3.000000  -1.000000
    abs_error_min             1.000000   0.000000
    rel_error_min            -0.500000  -2.000000
    abs_rel_error_min         0.125000   0.000000
    predicted_loa_lower       0.939302 -10.787363
    reference_loa_lower       1.765916 -12.995117
    error_loa_lower          -6.198064 -10.313018
    abs_error_loa_lower      -0.030349  -9.099293
    rel_error_loa_lower      -1.436945  -2.495168
    abs_rel_error_loa_lower  -0.492111  -1.256528
    predicted_loa_upper      11.060698  14.287363
    reference_loa_upper      10.234084   9.495117
    error_loa_upper           6.198064  17.313018
    abs_error_loa_upper       5.030349  17.099293
    rel_error_loa_upper       1.773254   1.678502
    abs_rel_error_loa_upper   1.614135   2.439861
    icc                       0.256198   0.328814
    icc_q05                  -0.710000  -0.670000
    icc_q95                   0.920000   0.940000
    n_common                  4.000000   4.000000
    n_additional_reference    0.000000   0.000000
    n_additional_predicted    0.000000   0.000000

    >>> pd.set_option("display.max_columns", None)
    >>> pd.set_option("display.width", 0)
    ...
    >>> predicted_sensor_left = pd.DataFrame(columns=["para"], data=[23, 82, 42]).rename_axis("s_id")
    >>> reference_sensor_left = pd.DataFrame(columns=["para"], data=[21, 86, 65]).rename_axis("s_id")
    >>> predicted_sensor_right = pd.DataFrame(columns=["para"], data=[26, -58, -3]).rename_axis("s_id")
    >>> reference_sensor_right = pd.DataFrame(columns=["para"], data=[96, -78, 86]).rename_axis("s_id")
    >>> calculate_aggregated_parameter_errors(
    ...     predicted_parameter={"left_sensor": predicted_sensor_left, "right_sensor": predicted_sensor_right},
    ...     reference_parameter={"left_sensor": reference_sensor_left, "right_sensor": reference_sensor_right},
    ...     id_column="s_id",
    ... )  # doctest: +NORMALIZE_WHITESPACE
                            left_sensor right_sensor
                                   para         para
    predicted_mean            49.000000   -11.666667
    reference_mean            57.333333    34.666667
    error_mean                -8.333333   -46.333333
    abs_error_mean             9.666667    59.666667
    rel_error_mean            -0.101707    -0.673487
    abs_rel_error_mean         0.165199     0.673487
    predicted_std             30.116441    42.665365
    reference_std             33.171273    97.700222
    error_std                 13.051181    58.226569
    abs_error_std             11.590226    35.641736
    rel_error_std              0.229574     0.392212
    abs_rel_error_std          0.165180     0.392212
    predicted_median          42.000000    -3.000000
    reference_median          65.000000    86.000000
    error_median              -4.000000   -70.000000
    abs_error_median           4.000000    70.000000
    rel_error_median          -0.046512    -0.729167
    abs_rel_error_median       0.095238     0.729167
    predicted_q05             24.900000   -52.500000
    reference_q05             25.400000   -61.600000
    error_q05                -21.100000   -87.100000
    abs_error_q05              2.200000    25.000000
    rel_error_q05             -0.323113    -1.004312
    abs_rel_error_q05          0.051384     0.303686
    predicted_q95             78.000000    23.100000
    reference_q95             83.900000    95.000000
    error_q95                  1.400000    11.000000
    abs_error_q95             21.100000    87.100000
    rel_error_q95              0.081063    -0.303686
    abs_rel_error_q95          0.327985     1.004312
    predicted_max             82.000000    26.000000
    reference_max             86.000000    96.000000
    error_max                  2.000000    20.000000
    abs_error_max             23.000000    89.000000
    rel_error_max              0.095238    -0.256410
    abs_rel_error_max          0.353846     1.034884
    predicted_min             23.000000   -58.000000
    reference_min             21.000000   -78.000000
    error_min                -23.000000   -89.000000
    abs_error_min              2.000000    20.000000
    rel_error_min             -0.353846    -1.034884
    abs_rel_error_min          0.046512     0.256410
    predicted_loa_lower      -10.028224   -95.290781
    reference_loa_lower       -7.682361  -156.825768
    error_loa_lower          -33.913649  -160.457409
    abs_error_loa_lower      -13.050176   -10.191136
    rel_error_loa_lower       -0.551671    -1.442223
    abs_rel_error_loa_lower   -0.158554    -0.095249
    predicted_loa_upper      108.028224    71.957448
    reference_loa_upper      122.349028   226.159101
    error_loa_upper           17.246982    67.790742
    abs_error_loa_upper       32.383509   129.524469
    rel_error_loa_upper        0.348258     0.095249
    abs_rel_error_loa_upper    0.488952     1.442223
    icc                        0.909121     0.628853
    icc_q05                    0.130000    -0.570000
    icc_q95                    1.000000     0.990000
    n_additional_predicted     0.000000     0.000000
    n_additional_reference     0.000000     0.000000
    n_common                   3.000000     3.000000

    >>> calculate_aggregated_parameter_errors(
    ...     predicted_parameter={"left_sensor": predicted_sensor_left, "right_sensor": predicted_sensor_right},
    ...     reference_parameter={"left_sensor": reference_sensor_left, "right_sensor": reference_sensor_right},
    ...     calculate_per_sensor=False
    ... )  # doctest: +NORMALIZE_WHITESPACE
                                   para
    predicted_mean            18.666667
    reference_mean            46.000000
    error_mean               -27.333333
    abs_error_mean            34.666667
    rel_error_mean            -0.387597
    abs_rel_error_mean         0.419343
    predicted_std             46.851539
    reference_std             66.425899
    error_std                 43.098337
    abs_error_std             36.219700
    rel_error_std              0.425081
    abs_rel_error_std          0.387238
    predicted_median          24.500000
    reference_median          75.500000
    error_median             -13.500000
    abs_error_median          21.500000
    rel_error_median          -0.305128
    abs_rel_error_median       0.305128
    predicted_q05            -44.250000
    reference_q05            -53.250000
    error_q05                -84.250000
    abs_error_q05              2.500000
    rel_error_q05             -0.958454
    abs_rel_error_q05          0.058693
    predicted_q95             72.000000
    reference_q95             93.500000
    error_q95                 15.500000
    abs_error_q95             84.250000
    rel_error_q95              0.059801
    abs_rel_error_q95          0.958454
    predicted_max             82.000000
    reference_max             96.000000
    error_max                 20.000000
    abs_error_max             89.000000
    rel_error_max              0.095238
    abs_rel_error_max          1.034884
    predicted_min            -58.000000
    reference_min            -78.000000
    error_min                -89.000000
    abs_error_min              2.000000
    rel_error_min             -1.034884
    abs_rel_error_min          0.046512
    predicted_loa_lower      -73.162349
    reference_loa_lower      -84.194761
    error_loa_lower         -111.806074
    abs_error_loa_lower      -36.323945
    rel_error_loa_lower       -1.220755
    abs_rel_error_loa_lower   -0.339643
    predicted_loa_upper      110.495682
    reference_loa_upper      176.194761
    error_loa_upper           57.139408
    abs_error_loa_upper      105.657279
    rel_error_loa_upper        0.445561
    abs_rel_error_loa_upper    1.178329
    icc                        0.663797
    icc_q05                   -0.090000
    icc_q95                    0.940000
    n_additional_predicted     0.000000
    n_additional_reference     0.000000
    n_common                   6.000000


    See Also
    --------
    gaitmap.parameters.TemporalParameterCalculation
    gaitmap.parameters.SpatialParameterCalculation

    Notes
    -----
    We are using pandas.quantile() to calculate the quantiles.
    In case the input is NaN/Inf for some values, the quantile function might return NaN/Inf as well.
    Even if the method returns a value, note, that pandas calculates the quantiles by first removing NaN and then
    passing them to `numpy.percentile()`.
    The result will be different from passing the data to `numpy.percentile()` directly.
    These NaN errors might happen for the relative errors, if the reference parameter is 0.

    """
    errors, common_rows_stats = calculate_parameter_errors(
        reference_parameter=reference_parameter, predicted_parameter=predicted_parameter, id_column=id_column
    )

    if isinstance(errors, pd.DataFrame):
        # This means we only had a single sensor in a DataFrame as input
        # Independent of the `calculate_per_sensor` parameter, we want to return a DataFrame
        return pd.concat([_calculate_error_stats(errors, scoring_errors=scoring_errors), common_rows_stats])
    if calculate_per_sensor is True:
        return pd.concat(
            {
                sensor_name: pd.concat(
                    [_calculate_error_stats(error_df, scoring_errors=scoring_errors), common_rows_stats[sensor_name]]
                )
                for sensor_name, error_df in errors.items()
            },
            axis=1,
        )
    # If we don't calculate per sensor, we combine the error dfs for all sensors
    combined_errors = pd.concat(errors)
    combined_errors.index = pd.Index(
        (f"{sensor_name}_{index}" for sensor_name, index in combined_errors.index), name=_ID_COL_NAME
    )

    # And we need to sum up the common rows stats
    common_rows_stats = pd.concat(common_rows_stats, axis=1).groupby(level=1, axis=1).sum()
    return pd.concat([_calculate_error_stats(combined_errors, scoring_errors=scoring_errors), common_rows_stats])


def _align_parameters(reference_parameter, predicted_parameter, id_column):
    sensor_names_list = sorted(set(predicted_parameter.keys()).intersection(reference_parameter.keys()))

    if len(sensor_names_list) == 0:
        raise ValidationError("The predicted values and the reference do not have any common sensors!")

    aligned_dict = {}
    meta_error_dict = {}

    for sensor in sensor_names_list:
        try:
            predicted_parameter_correct = set_correct_index(
                predicted_parameter[sensor], index_cols=[id_column]
            ).rename_axis(index={id_column: _ID_COL_NAME})
            reference_parameter_correct = set_correct_index(
                reference_parameter[sensor], index_cols=[id_column]
            ).rename_axis(index={id_column: _ID_COL_NAME})
        except ValidationError as e:
            raise ValidationError(
                f"Predicted and reference need to have either an index or a column named `{id_column}`. "
                "This column name is controlled by the `id_column` parameter.\n"
                "In case you are using the `parameter_pretty_` output of the parameter calculation, set this to "
                "`id_column='stride id'`."
            ) from e

        common_features = sorted(
            set(predicted_parameter_correct.keys()).intersection(reference_parameter_correct.keys())
        )

        err_msg_start = "No " if sensor == "__dummy__" else f"For sensor {sensor} no "

        if len(common_features) == 0:
            raise ValidationError(err_msg_start + "common parameter columns are found between predicted and reference.")

        aligned = pd.concat(
            [predicted_parameter_correct[common_features], reference_parameter_correct[common_features]],
            axis=1,
            keys=["predicted", "reference"],
            names=["source", "parameter"],
        )

        max_common = 0
        common_rows_per_parameter = {}
        for para in common_features:
            common = len(aligned.loc[:, pd.IndexSlice[:, para]].dropna(how="any"))
            max_common = max(common, max_common)
            common_rows_per_parameter[para] = {
                "n_common": common,
                "n_additional_reference": len(reference_parameter_correct[para].dropna()) - common,
                "n_additional_predicted": len(predicted_parameter_correct[para].dropna()) - common,
            }
        meta_error_dict[sensor] = pd.DataFrame(common_rows_per_parameter)

        if max_common == 0:
            raise ValidationError(err_msg_start + "common entries are found between predicted and reference!")

        aligned_dict[sensor] = aligned

    return aligned_dict, meta_error_dict


def _calculate_error(aligned_parameters: Dict[_Hashable, pd.DataFrame]) -> Dict[_Hashable, pd.DataFrame]:
    """Calculate the error between a reference and a predicted parameter."""
    final_error_dict = {}
    for k, v in aligned_parameters.items():
        error_dict = {
            "predicted": v["predicted"],
            "reference": v["reference"],
            "error": v["predicted"] - v["reference"],
            "abs_error": (v["predicted"] - v["reference"]).abs(),
            "rel_error": (v["predicted"] - v["reference"]) / v["reference"],
            "abs_rel_error": ((v["predicted"] - v["reference"]) / v["reference"]).abs(),
        }
        final_error_dict[k] = pd.concat(error_dict, axis=1, names=["error_type", "parameter"])
    return final_error_dict


def _calculate_error_stats(
    error_df: pd.DataFrame,
    scoring_errors: Literal["ignore", "warn", "raise"] = "warn",
) -> pd.DataFrame:
    """Aggregate the error for a single sensor."""
    assert {"error", "abs_error", "rel_error", "abs_rel_error", "predicted", "reference"} == set(
        error_df.columns.get_level_values(0)
    )
    assert error_df.columns.names == ["error_type", "parameter"]
    general_stats = (
        pd.DataFrame(  # noqa: PD010
            {
                "mean": error_df.mean(),
                "std": error_df.std(),
                "median": error_df.median(),
                "q05": error_df.quantile(0.05),
                "q95": error_df.quantile(0.95),
                "max": error_df.max(),
                "min": error_df.min(),
            }
        )
        .assign(loa_lower=lambda x: x["mean"] - 1.96 * x["std"], loa_upper=lambda x: x["mean"] + 1.96 * x["std"])
        .unstack("error_type")
        .T
    )
    general_stats.index = [f"{x[1]}_{x[0]}" for x in general_stats.index]

    # Finally we calculate the icc using the "predicted" and "reference" columns.
    icc = _icc(error_df[["predicted", "reference"]], scoring_errors=scoring_errors)
    if icc is not None:
        general_stats = pd.concat([general_stats, icc.T])

    return general_stats


def _icc(data: pd.DataFrame, scoring_errors: Literal["ignore", "warn", "raise"]):
    """Calculate the intraclass correlation coefficient using pingouin."""
    try:
        import pingouin as pg  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        if scoring_errors == "warn":
            warnings.warn(
                "The package pingouin is not installed. Calculating the intraclass correlation coefficient is skipped."
            )
        if scoring_errors == "raise":
            raise ImportError(
                "The package pingouin is not installed. Calculating the intraclass correlation coefficient is skipped."
            ) from e
        return None

    assert set(data.columns.get_level_values("error_type")) == {"predicted", "reference"}
    # If it is not unique, the ICC calculation will fail.
    assert data.index.is_unique

    paras = data.columns.get_level_values("parameter").unique()
    data = data.stack("error_type").reset_index()
    coefs: Dict[str, pd.Series] = {}
    for para in paras:
        try:
            # If handle error is ignore, we also ignore all warnings here.
            with warnings.catch_warnings():
                if scoring_errors == "ignore":
                    warnings.simplefilter("ignore")
                icc, ci95 = pg.intraclass_corr(
                    data, ratings=para, raters="error_type", targets=_ID_COL_NAME, nan_policy="omit"
                ).loc[0, ["ICC", "CI95%"]]
            coefs[para] = pd.Series({"icc": icc, "icc_q05": ci95[0], "icc_q95": ci95[1]})
        except AssertionError as e:
            if scoring_errors == "raise":
                raise
            if scoring_errors == "warn":
                warnings.warn(f"Calculating the intraclass correlation coefficient for {para} failed\n: {e}")
            coefs[para] = pd.Series({"icc": np.nan, "icc_q05": np.nan, "icc_q95": np.nan})

    return pd.concat(coefs, axis=1).T
