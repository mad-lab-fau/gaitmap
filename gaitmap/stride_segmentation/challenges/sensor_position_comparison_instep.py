"""A challenge for stride segmentation on long @lab gait tests of healthy participants.

General Information
-------------------

Dataset
    The `Sensor Position Comparison 2019 <dataset_info_>`_ dataset [1]_
    (`usage example <datasets_example_>`_, `download <dataset_download_>`_) contains 4x10m, 2x20m and a long walk
    (5 min) gait tests of 14 participants.
    The 4x10m and 2x20m tests are performed at 3 different speeds (slow, normal, fast).
    However, for this challenge, we consider all tests together as a single recording.
    This recording also contains potential walking and shuffling of the participants in between the tests.
    The dataset uses sensors at 6 different positions of each foot.
    However, for this challenge only the instep sensors are used.
Sensor System
    The instep sensors are NilsPod IMU sensors (204.8 Hz).
    The two sensors are synchronized with each other and the motion capture system with sub-sample accuracy.
Reference System
    The stride borders within the raw IMU data was labeled manually by an expert using the stride definition by [2]_.

Implementation Recommendations
------------------------------
For each participant, we only have a single recording that contains all tests.
This recording also contains all walking and movement between the tests.
Before each test, the participant was also jumping 3 times in place as a marker for the start of the test.
Further, some participants had to do some tests twice, because they did not follow the instructions.
All of this is included in the recording and all strides are labeled.

This means the algorithms should search for all strides (straight and turns) in the recording independent of everything.

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
from functools import partial
from pathlib import Path
from typing import ClassVar, Literal

import pandas as pd
from gaitmap_datasets.sensor_position_comparison_2019 import SensorPositionComparison2019Segmentation
from sklearn.model_selection import BaseCrossValidator
from tpcp import Pipeline
from tpcp.optimize import BaseOptimize
from tpcp.validate import cross_validate

from gaitmap.stride_segmentation.evaluation import (
    SingleValuePrecisionRecallF1,
    evaluate_segmented_stride_list,
    precision_recall_f1_score,
)
from gaitmap.utils.challenges import BaseChallenge, resolve_dataset

SensorNames = Literal["left_sensor", "right_sensor"]


def _get_data_subset(
    datapoint_data: pd.DataFrame | dict[str, pd.DataFrame], sensor="instep"
) -> dict[SensorNames, pd.DataFrame]:
    return {
        "left_sensor": datapoint_data[f"l_{sensor}"],
        "right_sensor": datapoint_data[f"r_{sensor}"],
    }


def final_scorer(
    pipeline: Pipeline,
    datapoint: SensorPositionComparison2019Segmentation,
    tolerance_s: float = 0.03,
    sensor_pos: str = "instep",
):
    """Score a pipeline build for segmentation challenges of the SensorPositionComparison2019 dataset.

    It compares the manually annotated reference borders from ``segmented_stride_list_per_sensor_``
    for the selected sensor position with the pipeline's ``stride_list_``.

    It calculates the precision, recall and f1 score for the stride segmentation.

    Parameters
    ----------
    pipeline
        The pipeline to score.
        The pipeline needs to have a `stride_list_` attribute that contains the segmented stride list after running.
    datapoint
        One recording from the SensorPositionComparison2019Segmentation dataset.
    tolerance_s
        The tolerance in seconds that is allowed between the calculated stride borders and the reference stride
        borders.
        Both, start and end labels need to be within the tolerance to be considered a match.
    sensor_pos
        The sensor position to use for the scoring.
        This will not affect the evaluation, as the dataset stores the same ground truth for all sensor positions,
        but we include the active selection of the sensor position for completeness.

    See Also
    --------
    gaitmap.stride_segmentation.evaluation.evaluate_segmented_stride_list

    """
    results = pipeline.safe_run(datapoint)

    assert hasattr(results, "stride_list_"), "The pipeline must provide its results as a `stride_list_` attribute."

    matched_stride_list = evaluate_segmented_stride_list(
        ground_truth=_get_data_subset(datapoint.segmented_stride_list_per_sensor_, sensor=sensor_pos),
        segmented_stride_list=results.stride_list_,
        tolerance=int(tolerance_s * datapoint.sampling_rate_hz),
    )

    combined_matched_stride_list = pd.concat(matched_stride_list)
    return {
        **precision_recall_f1_score(combined_matched_stride_list),
        "per_sample": SingleValuePrecisionRecallF1(return_raw_scores=False)(combined_matched_stride_list),
    }


ChallengeDataset = SensorPositionComparison2019Segmentation


@dataclass(repr=False)
class Challenge(BaseChallenge):
    """The SensorPositionComparison Stride Segmentaion Validation Challenge.

    Parameters
    ----------
    dataset
        An instance of :class:`~gaitmap_datasets.SensorPositionComparison2019Segmentation` or a directory containing
        the dataset.
    cv_iterator
        A cross-validation iterator or the number of folds to use.
    cv_params
        Additional parameters to pass to the tpcp cross-validation function.

    Other Parameters
    ----------------
    match_tolerance_s
        (Class Constant) The tolerance in seconds that is allowed between the calculated stride borders and the
        reference stride borders.
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
    match_tolerance_s: ClassVar[float] = 0.03
    sensor_pos: ClassVar[str] = "instep"

    # Update the version, when the challenge_class is changed in a relevant way
    VERSION = "1.0.0"

    cv_results_: dict = field(init=False)

    def run(self, optimizer: BaseOptimize):
        """Run cross-validation with the prescribed scorer and dataset splits."""
        with self._measure_time():
            self.optimizer = optimizer
            self.dataset_ = self._resolve_dataset()
            cv_params = {} if self.cv_params is None else self.cv_params

            self.cv_results_ = cross_validate(
                optimizable=optimizer,
                dataset=self.dataset_,
                cv=self._get_cv_iterator(),
                scoring=self.get_scorer(),
                return_optimizer=True,
                **cv_params,
            )
        return self

    def _resolve_dataset(self):
        return resolve_dataset(self.dataset, ChallengeDataset)

    @classmethod
    def get_scorer(cls):
        """Return the scoring function prescribed by this challenge."""
        return partial(final_scorer, tolerance_s=cls.match_tolerance_s, sensor_pos=cls.sensor_pos)

    @classmethod
    def get_imu_data(cls, datapoint: ChallengeDataset) -> dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        """Select the IMU input required by this challenge."""
        return _get_data_subset(datapoint.data, sensor=cls.sensor_pos)

    @classmethod
    def get_reference_stride_list(
        cls,
        datapoint: ChallengeDataset,
    ) -> dict[Literal["left_sensor", "right_sensor"], pd.DataFrame]:
        """Return the reference stride borders used for scoring."""
        return _get_data_subset(datapoint.segmented_stride_list_per_sensor_, sensor=cls.sensor_pos)


__all__ = ["Challenge", "ChallengeDataset", "SensorNames", "final_scorer"]
