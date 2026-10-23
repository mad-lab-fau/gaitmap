"""Evaluation helpers for full gait pipelines."""

from collections.abc import Sequence
from typing import Literal

import pandas as pd
from tpcp.validate import Aggregator

from gaitmap.parameters.evaluation import calculate_aggregated_parameter_errors, calculate_parameter_errors

_AggType = dict[Literal["reference", "predicted"], pd.Series]


class ParameterErrors(Aggregator[_AggType]):
    """Compare recording-level parameter estimates across a dataset."""

    def aggregate(self, /, values: Sequence[_AggType], datapoints) -> dict[str, float]:
        """Aggregate predictions from the supplied recordings."""
        all_references = pd.DataFrame(
            {str(dp.group_label): val["reference"] for dp, val in zip(datapoints, values, strict=False)}
        ).T
        all_predictions = pd.DataFrame(
            {str(dp.group_label): val["predicted"] for dp, val in zip(datapoints, values, strict=False)}
        ).T

        all_references.index.name = "group"
        all_predictions.index.name = "group"

        errors = calculate_aggregated_parameter_errors(
            reference_parameter=all_references,
            predicted_parameter=all_predictions,
            calculate_per_sensor=True,
            scoring_errors="ignore",
            id_column="group",
        ).stack()
        errors.index = [f"{c[1]}__{c[0]}" for c in errors.index]

        return errors.to_dict()


__all__ = ["ParameterErrors", "calculate_aggregated_parameter_errors", "calculate_parameter_errors"]
