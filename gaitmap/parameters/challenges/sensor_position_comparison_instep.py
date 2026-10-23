"""A challenge for stride-by-stride comparison of spatial parameters on various lab tests.

General Information
-------------------

Dataset
    The `Sensor Position Comparison 2019 <dataset_info_>`_ dataset [1]_
    (`usage example <datasets_example_>`_, `download <dataset_download_>`_) contains 4x10m, 2x20m and a long walk
    (5 min) gait tests of 14 participants.
    The 4x10m and 2x20m tests are performed at 3 different speeds (slow, normal, fast).
    The dataset uses sensors at 6 different positions of each foot.
    However, for this challenge only the instep sensors are used.
Sensor System
    The instep sensors are NilsPod IMU sensors (204.8 Hz).
    The two sensors are synchronized with each other and the motion capture system with sub-sample accuracy.
Reference System
    A marker-based motion capture system by Qualisys (Opus 700+ Qualisys, 28 cameras, 20x30 m capture volume) at 100 Hz
    is used to track the foot position using 4 markers per foot (at the calcaneus (CAL), at the tip of the shoe (TOE),
    and on top of the first and the fifth metatarsal (MET1 and MET5).
    For this challenge the marker at the calcaneus is used to calculate stride length.

Implementation Recommendations
------------------------------
The pipeline should use the raw data and the segmented stride list provided by the dataset.
The start and end value of each stride in the provided segmented stride list follow the stride definitions by Barth et
gal. [2]_.
This means the start and end-values are defined by the minimum in the gyr_ml axis right before the toe-off.
The ground truth stride length is calculated from the marker position at the calcaneus (CAL) by calculating the distance
traveled by this marker in the ground plane between the heel strike right before a stride and the heel strike within
the stride.
This means for the first stride in the segmented stride list and for each stride after a break, no ground truth stride
length can be calculated.
The pipeline should also not calculate parameters for these strides.
To correctly handle these shifts in stride definition, you can use
:func:`~gaitmap.utils.stride_list_conversion.convert_stride_list` or check the stride ids in the calculated
ground truth parameters to remove strides without ground truth from your calculations.
Further check the `dataset example <datasets_example_>`_ for more guidance on this.
The final calculated parameters should match the provided stride list and should have the same stride ids.
Note, that we assume that parameters are calculated for each stride.
Missing strides are not handled by the evaluation.

The strides also include turning strides.
Your method should be prepared to handle them.

References
----------
.. [1] Küderle, Arne, Nils Roth, Jovana Zlatanovic, Markus Zrenner, Bjoern Eskofier, and Felix Kluge. “The Placement of
   Foot-Mounted IMU Sensors Does Affect the Accuracy of Spatial Parameters during Regular Walking.” PLOS ONE 17, no. 6
   (June 9, 2022): e0269567. https://doi.org/10.1371/journal.pone.0269567.
.. [2] Barth, Jens, Cäcilia Oberndorfer, Cristian Pasluosta, Samuel Schülein, Heiko Gassner, Samuel Reinfelder,
   Patrick Kugler, et al. “Stride Segmentation during Free Walk Movements Using Multi-Dimensional Subsequence Dynamic
   Time Warping on Inertial Sensor Data.” Sensors (Switzerland) 15, no. 3 (March 17, 2015): 6419-40.
   https://doi.org/10.3390/s150306419.


.. _dataset_info: https://zenodo.org/record/5747173
.. _datasets_example: https://mad-lab-fau.github.io/gaitmap-datasets/auto_examples/sensor_position_comparison_2019.html
.. _dataset_download: https://zenodo.org/record/5747173

"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    ClassVar,
    Literal,
)

import pandas as pd
from gaitmap_datasets.sensor_position_comparison_2019 import (
    SensorPositionComparison2019Mocap,
)
from sklearn.model_selection import BaseCrossValidator
from tpcp import Pipeline
from tpcp.optimize import BaseOptimize
from tpcp.validate import DatasetSplitter, cross_validate, no_agg

from gaitmap.parameters import SpatialParameterCalculation, TemporalParameterCalculation
from gaitmap.parameters.evaluation import SingleValueErrors, calculate_aggregated_parameter_errors
from gaitmap.utils.challenges import BaseChallenge
from gaitmap.utils.datatype_helper import set_correct_index
from gaitmap.utils.stride_list_conversion import convert_stride_list

ChallengeDataset = SensorPositionComparison2019Mocap
SensorNames = Literal["left_sensor", "right_sensor"]


def _get_data_subset(
    datapoint_data: pd.DataFrame | dict[str, pd.DataFrame], sensor="instep"
) -> dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
    return {
        "left_sensor": datapoint_data[f"l_{sensor}"],
        "right_sensor": datapoint_data[f"r_{sensor}"],
    }


def final_scorer(pipeline: Pipeline, datapoint: ChallengeDataset):
    """Score a pipeline build for the SensorPositionComparison Spatial Parameter challenge on a single datapoint.

    It compares the stride length on a stride-by-stride bases calculating common error metrics.

    Parameters
    ----------
    pipeline
        The pipeline to score.
        This is expected to have the attribute `parameters_` after running.
        It should contain the calculated parameters for each stride.
    datapoint
        A datapoint of the SensorPositionComparison2019Mocap dataset.

    """
    results = pipeline.safe_run(datapoint)
    predicted = {k: v[["stride_length"]] for k, v in results.parameters_.items()}
    # Calling the Challenge method here is not optimal, as it couples this function to the Challenge class.
    # But we keep it for now, until it creates a problem.
    reference = {k: v[["stride_length"]] for k, v in Challenge.get_ground_truth_parameter(datapoint).items()}

    errors = calculate_aggregated_parameter_errors(
        reference_parameter=reference,
        predicted_parameter=predicted,
        calculate_per_sensor=False,
        scoring_errors="ignore",
    )["stride_length"]

    return {
        **errors.to_dict(),
        "per_stride": SingleValueErrors(return_raw_scores=False)(
            ("stride_length", {"reference": reference, "predicted": predicted})
        ),
        "stride_length_predicted": no_agg(
            pd.concat(
                {k: set_correct_index(v, ["s_id"]) for k, v in predicted.items()},
                axis=0,
            )["stride_length"]
        ),
        "stride_length_reference": no_agg(
            pd.concat(
                {k: set_correct_index(v, ["s_id"]) for k, v in reference.items()},
                axis=0,
            )["stride_length"]
        ),
    }


@dataclass(repr=False)
class Challenge(BaseChallenge):
    """The SensorPositionComparison Spatial Parameter Challenge.

    Parameters
    ----------
    dataset
        A instance of :class:`~gaitmap_datasets.SensorPositionComparison2019Mocap` or a path to a directory containing
        the dataset.
    cv_iterator
        A cross-validation iterator or the number of folds to use.
    cv_params
        Additional parameters to pass to the tpcp cross-validation function.

    Other Parameters
    ----------------
    ground_truth_marker
        (Class Constant) The marker used to calculate the ground truth stride borders.
    data_padding_s
        (Class Constant) The amount of padding in seconds to add before and after each gait tests.
        This ensures that sufficient resting data is available before and after the gait test.
    sensor_pos
        (Class Constant) The sensor position to use for the challenge.

    Attributes
    ----------
    cv_results_
        The results of the cross-validation.
        This can be passed directly to the pandas DataFrame constructor to get a dataframe with the results.

    See Also
    --------
    gaitmap.utils.challenges.BaseChallenge : For common parameters and attributes of all challenges.

    """

    dataset: str | Path | ChallengeDataset | None
    cv_iterator: int | BaseCrossValidator | Iterator | None = 5
    cv_params: dict | None = None

    # Class level config.
    sensor_pos: ClassVar[str] = "instep"
    ground_truth_marker: ClassVar[Literal["toe", "fcc", "fm1", "fm5"]] = "fcc"
    data_padding_s: ClassVar[float] = 3.0

    # Update the version, when the challenge_class is changed in a relevant way
    VERSION = "1.0.0"

    cv_results_: dict = field(init=False)

    def run(self, optimizer: BaseOptimize):
        """Run cross-validation with the prescribed scorer and dataset splits."""
        with self._measure_time():
            self.optimizer = optimizer
            self.dataset_ = self._resolve_dataset()
            self.dataset_ = self.dataset_.clone().set_params(data_padding_s=self.data_padding_s)
            cv_params = {} if self.cv_params is None else self.cv_params

            self.cv_results_ = cross_validate(
                optimizable=optimizer,
                dataset=self.dataset_,
                cv=DatasetSplitter(self._get_cv_iterator(), groupby=["participant"], stratify=None),
                scoring=self.get_scorer(),
                return_optimizer=True,
                **cv_params,
            )
        return self

    def _resolve_dataset(self):
        if isinstance(self.dataset, (str, Path)):
            return ChallengeDataset(data_folder=Path(self.dataset))
        if isinstance(self.dataset, ChallengeDataset):
            return self.dataset
        raise ValueError(
            "`dataset` must either be a valid path or a valid instance of `SensorPositionComparison2019Mocap`."
        )

    @classmethod
    def get_scorer(cls):
        """Return the scoring function prescribed by this challenge."""
        return final_scorer

    @classmethod
    def get_imu_data(cls, datapoint: ChallengeDataset) -> dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        """Select the IMU input required by this challenge."""
        return _get_data_subset(datapoint.data, sensor=cls.sensor_pos)

    @classmethod
    def get_ground_truth_segmented_stride_list(
        cls,
        datapoint: ChallengeDataset,
    ) -> dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        """Return the reference stride borders for parameter estimation."""
        return _get_data_subset(datapoint.segmented_stride_list_per_sensor_, sensor=cls.sensor_pos)

    @classmethod
    def get_ground_truth_parameter(cls, datapoint: ChallengeDataset):
        """Calculate reference parameters from the reference system."""
        events = convert_stride_list(datapoint.mocap_events_, target_stride_type="ic")
        events = {f"{k}_sensor": v for k, v in events.items()}

        tp = TemporalParameterCalculation(expected_stride_type="ic").calculate(
            stride_event_list=events,
            sampling_rate_hz=datapoint.mocap_sampling_rate_hz_,
        )

        marker_positions = {
            f"{foot}_sensor": datapoint.marker_position_per_stride_[foot][
                f"{foot[0]}_{cls.ground_truth_marker}"
            ].rename(columns=lambda x: f"pos_{x}")
            for foot in ["left", "right"]
        }

        sp = SpatialParameterCalculation(expected_stride_type="ic").calculate(
            positions=marker_positions,
            orientations=None,
            stride_event_list=events,
            sampling_rate_hz=datapoint.mocap_sampling_rate_hz_,
        )

        return {k: pd.concat([tp.parameters_[k], sp.parameters_[k]], axis=1) for k in ["left_sensor", "right_sensor"]}


__all__ = ["Challenge", "ChallengeDataset", "final_scorer"]
