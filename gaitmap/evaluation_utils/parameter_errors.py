"""A helper function to evaluate the output of the temporal or spatial parameter calculation against a ground truth."""
import warnings
from typing import Dict, Literal, Union

import numpy as np
import pandas as pd

from gaitmap.utils._types import _Hashable
from gaitmap.utils.datatype_helper import set_correct_index
from gaitmap.utils.exceptions import ValidationError


def calculate_parameter_errors(
    *,
    reference_parameter: Union[pd.DataFrame, Dict[_Hashable, pd.DataFrame]],
    predicted_parameter: Union[pd.DataFrame, Dict[_Hashable, pd.DataFrame]],
    calculate_per_sensor: bool = True,
    scoring_errors: Literal["ignore", "warn", "raise"] = "warn",
) -> pd.DataFrame:
    """Calculate various error metrics between a parameter predicted and a given ground truth.

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

    In addition, the number of common strides, additional strides in the reference and additional strides in the
    predicted values are calculated.
    These metrics are helpful, as parameter errors are only calculated for strides that are present in both the
    inputs.
    Strides between the predicted and the reference are matched based on the `s_id`/`stride id` column/index.
    Parameters with `np.nan` are considered to be missing for the respective stride.

    The metrics can either be calculated per sensor or for all sensors combined (see the `calculate_per_sensor`
    parameter).
    In the latter case, the error per stride is calculated and then all strides of all sensors combined before
    calculating the summary metrics (mean, std, ...).
    This might be desired, if you have one sensor per foot, but want to have statistics over all strides independent of
    the foot.

    Parameters
    ----------
    reference_parameter
        The reference the predicted values should be compared against.
        This must be the same type (i.e. single/multi sensor) as the predicted input.
        Further, sensor names, column names, and stride ids must match with the `predicted_parameters`.
    predicted_parameter
        The output of the temporal or spatial parameter calculation (both `.parameters_` and `.parameters_pretty_`
        are accepted).
        This can be a Dataframe or a dict of such Dataframes.
    calculate_per_sensor
        A bool that can be set to `False` if you wish to calculate error metrics as if the
        strides were all taken by one sensor.
        Default is `True`.
    scoring_errors
        How to handle errors during the scoring.
        Can be one of `ignore`, `warn`, or `raise`.
        Default is `warn`.
        At the moment, this only effects the calculation of the ICC.
        In all cases the value for a given metric is set to `np.nan`.

    Returns
    -------
    output
        A Dataframe with one row per error metric and one column per parameter.
        In case of a multi-sensor predicted (and `calculate_per_sensor=True`), the dataframe has 2 column levels.
        The first level is the sensor name and the second one the parameter name.

    Examples
    --------
    >>> predicted_param = pd.DataFrame({"para1": [7, 3, 5, 9], "para2": [7, -1, 7, -6]}).rename_axis("stride id")
    >>> reference = pd.DataFrame({"para1": [3, 6, 7, 8], "para2": [-7, -1, 6, -5]}).rename_axis("stride id")
    >>> calculate_parameter_errors(
    ...     predicted_parameter=predicted_param,
    ...     reference_parameter=reference,
    ... )  # doctest: +NORMALIZE_WHITESPACE
                                    para1      para2
    error_mean               0.000000   3.500000
    error_std                3.162278   7.047458
    error_median            -0.500000   0.500000
    error_q05               -2.850000  -0.850000
    error_q95                3.550000  12.050000
    error_max                4.000000  14.000000
    error_min               -3.000000  -1.000000
    error_loa_lower         -6.198064 -10.313018
    error_loa_upper          6.198064  17.313018
    rel_error_mean           0.168155   0.491667
    rel_error_std            0.818928   1.016667
    rel_error_median        -0.080357   0.083333
    rel_error_q05           -0.467857  -0.170000
    rel_error_q95            1.152083   1.725000
    rel_error_max            1.333333   2.000000
    rel_error_min           -0.500000  -0.200000
    rel_error_loa_lower     -1.436945  -1.501000
    rel_error_loa_upper      1.773254   2.484333
    abs_error_mean           2.500000   4.000000
    abs_error_std            1.290994   6.683313
    abs_error_median         2.500000   1.000000
    abs_error_q05            1.150000   0.150000
    abs_error_q95            3.850000  12.050000
    abs_error_max            4.000000  14.000000
    abs_error_min            1.000000   0.000000
    abs_error_loa_lower     -0.030349  -9.099293
    abs_error_loa_upper      5.030349  17.099293
    abs_rel_error_mean       0.561012   0.591667
    abs_rel_error_std        0.537307   0.942956
    abs_rel_error_median     0.392857   0.183333
    abs_rel_error_q05        0.149107   0.025000
    abs_rel_error_q95        1.208333   1.730000
    abs_rel_error_max        1.333333   2.000000
    abs_rel_error_min        0.125000   0.000000
    abs_rel_error_loa_lower -0.492111  -1.256528
    abs_rel_error_loa_upper  1.614135   2.439861
    n_common                 4.000000   4.000000
    n_additional_reference   0.000000   0.000000
    n_additional_predicted   0.000000   0.000000
    icc                      0.256198   0.328814
    icc_q05                 -0.710000  -0.670000
    icc_q95                  0.920000   0.940000

    >>> pd.set_option("display.max_columns", None)
    >>> pd.set_option("display.width", 0)
    ...
    >>> predicted_sensor_left = pd.DataFrame(columns=["para"], data=[23, 82, 42]).rename_axis("s_id")
    >>> reference_sensor_left = pd.DataFrame(columns=["para"], data=[21, 86, 65]).rename_axis("s_id")
    >>> predicted_sensor_right = pd.DataFrame(columns=["para"], data=[26, -58, -3]).rename_axis("s_id")
    >>> reference_sensor_right = pd.DataFrame(columns=["para"], data=[96, -78, 86]).rename_axis("s_id")
    >>> calculate_parameter_errors(
    ...     predicted_parameter={"left_sensor": predicted_sensor_left, "right_sensor": predicted_sensor_right},
    ...     reference_parameter={"left_sensor": reference_sensor_left, "right_sensor": reference_sensor_right}
    ... )  # doctest: +NORMALIZE_WHITESPACE
                            left_sensor right_sensor
                                   para         para
    error_mean                -8.333333   -46.333333
    error_std                 13.051181    58.226569
    error_median              -4.000000   -70.000000
    error_q05                -21.100000   -87.100000
    error_q95                  1.400000    11.000000
    error_max                  2.000000    20.000000
    error_min                -23.000000   -89.000000
    error_loa_lower          -33.913649  -160.457409
    error_loa_upper           17.246982    67.790742
    rel_error_mean            -0.101707    -0.502547
    rel_error_std              0.229574     0.674817
    rel_error_median          -0.046512    -0.729167
    rel_error_q05             -0.323113    -1.004312
    rel_error_q95              0.081063     0.157853
    rel_error_max              0.095238     0.256410
    rel_error_min             -0.353846    -1.034884
    rel_error_loa_lower       -0.551671    -1.825187
    rel_error_loa_upper        0.348258     0.820094
    abs_error_mean             9.666667    59.666667
    abs_error_std             11.590226    35.641736
    abs_error_median           4.000000    70.000000
    abs_error_q05              2.200000    25.000000
    abs_error_q95             21.100000    87.100000
    abs_error_max             23.000000    89.000000
    abs_error_min              2.000000    20.000000
    abs_error_loa_lower      -13.050176   -10.191136
    abs_error_loa_upper       32.383509   129.524469
    abs_rel_error_mean         0.165199     0.673487
    abs_rel_error_std          0.165180     0.392212
    abs_rel_error_median       0.095238     0.729167
    abs_rel_error_q05          0.051384     0.303686
    abs_rel_error_q95          0.327985     1.004312
    abs_rel_error_max          0.353846     1.034884
    abs_rel_error_min          0.046512     0.256410
    abs_rel_error_loa_lower   -0.158554    -0.095249
    abs_rel_error_loa_upper    0.488952     1.442223
    n_common                   3.000000     3.000000
    n_additional_reference     0.000000     0.000000
    n_additional_predicted     0.000000     0.000000
    icc                        0.909121     0.628853
    icc_q05                    0.130000    -0.570000
    icc_q95                    1.000000     0.990000

    >>> calculate_parameter_errors(
    ...     predicted_parameter={"left_sensor": predicted_sensor_left, "right_sensor": predicted_sensor_right},
    ...     reference_parameter={"left_sensor": reference_sensor_left, "right_sensor": reference_sensor_right},
    ...     calculate_per_sensor=False
    ... )  # doctest: +NORMALIZE_WHITESPACE
                                   para
    error_mean               -27.333333
    error_std                 43.098337
    error_median             -13.500000
    error_q05                -84.250000
    error_q95                 15.500000
    error_max                 20.000000
    error_min                -89.000000
    error_loa_lower         -111.806074
    error_loa_upper           57.139408
    rel_error_mean            -0.302127
    rel_error_std              0.501432
    rel_error_median          -0.200179
    rel_error_q05             -0.958454
    rel_error_q95              0.216117
    rel_error_max              0.256410
    rel_error_min             -1.034884
    rel_error_loa_lower       -1.284934
    rel_error_loa_upper        0.680681
    abs_error_mean            34.666667
    abs_error_std             36.219700
    abs_error_median          21.500000
    abs_error_q05              2.500000
    abs_error_q95             84.250000
    abs_error_max             89.000000
    abs_error_min              2.000000
    abs_error_loa_lower      -36.323945
    abs_error_loa_upper      105.657279
    abs_rel_error_mean         0.419343
    abs_rel_error_std          0.387238
    abs_rel_error_median       0.305128
    abs_rel_error_q05          0.058693
    abs_rel_error_q95          0.958454
    abs_rel_error_max          1.034884
    abs_rel_error_min          0.046512
    abs_rel_error_loa_lower   -0.339643
    abs_rel_error_loa_upper    1.178329
    n_common                   6.000000
    n_additional_reference     0.000000
    n_additional_predicted     0.000000
    icc                        0.110385
    icc_q05                   -0.860000
    icc_q95                    0.960000

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

    output = _calculate_error(reference_parameter, predicted_parameter, calculate_per_sensor, scoring_errors)

    if predicted_is_not_dict:
        output = output["__dummy__"]

    return output


def _calculate_error(  # noqa: C901
    reference_parameter: Union[pd.DataFrame, Dict[_Hashable, pd.DataFrame]],
    predicted_parameter: Union[pd.DataFrame, Dict[_Hashable, pd.DataFrame]],
    calculate_per_sensor: bool,
    scoring_errors: Literal["ignore", "warn", "raise"],
) -> pd.DataFrame:
    sensor_names_list = sorted(set(predicted_parameter.keys()).intersection(reference_parameter.keys()))

    if len(sensor_names_list) == 0:
        raise ValidationError("The predicted values and the reference do not have any common sensors!")

    aligned_dict = {}
    meta_error_dict = {}

    for sensor in sensor_names_list:
        try:
            predicted_parameter_correct = set_correct_index(predicted_parameter[sensor], index_cols=["s_id"])
            reference_parameter_correct = set_correct_index(reference_parameter[sensor], index_cols=["s_id"])
        except ValidationError:
            try:
                predicted_parameter_correct = set_correct_index(
                    predicted_parameter[sensor], index_cols=["stride id"]
                ).rename_axis(index={"stride id": "s_id"})
                reference_parameter_correct = set_correct_index(
                    reference_parameter[sensor], index_cols=["stride id"]
                ).rename_axis(index={"stride id": "s_id"})
            except ValidationError as e:
                raise ValidationError(
                    'Predicted and reference need to have either an index or a column named "s_id" or "stride id". '
                    "Note, that predicted and reference must both use the same name for the id-column."
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
        for para in common_features:
            common = len(aligned.loc[:, pd.IndexSlice[:, para]].dropna(how="any"))
            max_common = max(common, max_common)
            meta_error_dict[(sensor, para)] = {
                "n_common": common,
                "n_additional_reference": len(reference_parameter_correct[para].dropna()) - common,
                "n_additional_predicted": len(predicted_parameter_correct[para].dropna()) - common,
            }

        if max_common == 0:
            raise ValidationError(err_msg_start + "common strides are found between predicted and reference!")

        aligned_dict[sensor] = aligned

    meta_error_df = pd.DataFrame(meta_error_dict).T

    if calculate_per_sensor is True:
        error_df = {
            k: _error_single_df(v, meta_error_df.loc[k], handle_error=scoring_errors) for k, v in aligned_dict.items()
        }
        return pd.concat(error_df, axis=1)
    return _error_single_df(
        pd.concat(aligned_dict.values()), meta_error_df.groupby(level=1, axis=0).sum(), handle_error=scoring_errors
    )


def _error_single_df(
    df: pd.DataFrame, meta_error: pd.DataFrame, handle_error: Literal["ignore", "warn", "raise"]
) -> pd.DataFrame:
    error = df["predicted"] - df["reference"]
    output = [
        _max_mean_median_std_quantille(error).add_prefix("error_"),
        _max_mean_median_std_quantille(error / df["reference"].abs()).add_prefix("rel_error_"),
        _max_mean_median_std_quantille(error.abs()).add_prefix("abs_error_"),
        _max_mean_median_std_quantille(error.abs() / df["reference"].abs()).add_prefix("abs_rel_error_"),
        meta_error,
    ]
    icc = _icc(df, handle_error=handle_error)
    if icc is not None:
        output.append(icc)

    return pd.concat(output, axis=1).T


def _max_mean_median_std_quantille(value: pd.DataFrame) -> pd.DataFrame:
    """Calculate the mean, median, standard deviation and the 5%/95% quantille of the input dataframe per column."""
    return pd.DataFrame(
        {
            "mean": value.mean(),
            "std": value.std(),
            "median": value.median(),
            "q05": value.quantile(0.05),
            "q95": value.quantile(0.95),
            "max": value.max(),
            "min": value.min(),
        }
    ).assign(loa_lower=lambda x: x["mean"] - 1.96 * x["std"], loa_upper=lambda x: x["mean"] + 1.96 * x["std"])


def _icc(data: pd.DataFrame, handle_error: Literal["ignore", "warn", "raise"]):
    """Calculate the intraclass correlation coefficient using pingouin."""
    try:
        import pingouin as pg  # pylint: disable=import-outside-toplevel
    except ImportError:
        warnings.warn(
            "The package pingouin is not installed. Calculating the intraclass correlation coefficient is skipped."
        )
        return None
    data = data.stack("source").reset_index()
    coefs: Dict[str, pd.Series] = {}
    paras = set(data.columns) - {"source", "s_id"}
    for para in paras:
        try:
            icc, ci95 = pg.intraclass_corr(data, ratings=para, raters="source", targets="s_id", nan_policy="omit").loc[
                0, ["ICC", "CI95%"]
            ]
            coefs[para] = pd.Series({"icc": icc, "icc_q05": ci95[0], "icc_q95": ci95[1]})
        except AssertionError as e:
            if handle_error == "raise":
                raise
            if handle_error == "warn":
                warnings.warn(f"Calculating the intraclass correlation coefficient for {para} failed\n: {e}")
            coefs[para] = pd.Series({"icc": np.nan, "icc_q05": np.nan, "icc_q95": np.nan})

    return pd.concat(coefs, axis=1).T
