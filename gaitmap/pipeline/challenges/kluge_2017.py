"""A challenge to test the performance of a full gaitanalysis pipeline on laboratory 4x10m gait tests.

Comparisons are performed on the calculated means of spatial temporal parameters over the entire gait test.
Reference gait parameters are provided by a marker-less motion capture system that covers the center of the walking
section.
The entire validation is run as a 5-fold cross-validation to allow the algorithms to optimize parameters on an
independent train set.

General Information
-------------------

Dataset
    The `Kluge 2017 <dataset_info_>`_  dataset [1]_
    (`usage example <datasets_example_>`_, `download <dataset_download_>`_) contains laboratory 4x10m gait tests.
Sensor System
    Two IMU sensors (Shimmer 3, 102.4 Hz) are attached laterally to the shoes of the participants.
Reference System
    A marker-less motion capture system (Simi Motion, 100 Hz) is used to track the foot and ankle trajectory.
    For this challenge, we use the trajectory of the ankle marker to calculate stride length.
    Heel strikes were labeled manually based on the video data.

Implementation Recommendations
------------------------------
As the MoCap system does not cover the turns at the end of each 10m section algorithms should also cut of all turning
strides (recommendation: turning angle > 20 deg) before calculating the mean and the variance.
Otherwise, calculated results will have a considerable bias.

Notes
-----
The comparison here is fundamentally different that the validation performed in the original paper [1]_.
There, parameters were compared on a per-stride basis, while here the mean over the entire gait test is used.
Hence, error values are not directly comparable.

References
----------
.. [1] Kluge, Felix, Heiko Gaßner, Julius Hannink, Cristian Pasluosta, Jochen Klucken, and Björn M. Eskofier.
     “Towards Mobile Gait Analysis: Concurrent Validity and Test-Retest Reliability of an Inertial Measurement System
     for the Assessment of Spatio-Temporal Gait Parameters.” Sensors 17, no. 7 (July 2017): 1522.
     https://doi.org/10.3390/s17071522.

.. _dataset_info: https://www.mad.tf.fau.de/research/datasets/#collapse_13
.. _datasets_example: https://mad-lab-fau.github.io/gaitmap-datasets/auto_examples/kluge_2017.html
.. _dataset_download: https://osf.io/cfb7e/

"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from gaitmap_datasets import Kluge2017
from sklearn.model_selection import BaseCrossValidator, StratifiedGroupKFold
from tpcp import Pipeline
from tpcp.optimize import BaseOptimize
from tpcp.validate import DatasetSplitter, cross_validate, no_agg

from gaitmap.parameters import SpatialParameterCalculation, TemporalParameterCalculation
from gaitmap.parameters.evaluation import calculate_parameter_errors
from gaitmap.pipeline.evaluation import ParameterErrors
from gaitmap.utils.challenges import BaseChallenge, resolve_dataset
from gaitmap.utils.consts import GF_POS

__all__ = ["Challenge", "ChallengeDataset", "final_scorer"]

ChallengeDataset = Kluge2017


def final_scorer(pipeline: Pipeline, datapoint: ChallengeDataset):
    """Score a pipeline build for the Kluge2017 challenge on a single datapoint.

    It compares the mean gait parameters of the entire gait test between the pipeline and the reference.

    Parameters
    ----------
    pipeline
        The pipeline to score.
        This is expected to have the attribute `aggregated_gait_parameters_` after running.
        It should contain the aggregated gait parameters for the entire gait test.
    datapoint
        A datapoint of the Kluge2017 dataset.

    """
    results = pipeline.safe_run(datapoint)

    aggregated_paras = results.aggregated_gait_parameters_
    aggregated_ground_truth = Challenge.get_aggregated_ground_truth_parameter(datapoint)

    errors = calculate_parameter_errors(
        predicted_parameter=aggregated_paras.to_frame().T.reset_index(drop=True).rename_axis(index="test"),
        reference_parameter=aggregated_ground_truth.to_frame().T.reset_index(drop=True).rename_axis(index="test"),
        id_column="test",
    )[0].iloc[0]

    errors.index = [f"{parameter}__{metric}" for metric, parameter in errors.index]

    return {
        **{k: no_agg(v) for k, v in errors.to_dict().items()},
        "agg": ParameterErrors(return_raw_scores=False)(
            {"predicted": aggregated_paras, "reference": aggregated_ground_truth}
        ),
    }


@dataclass(repr=False)
class Challenge(BaseChallenge):
    """The Kluge2017 Challenge.

    Parameters
    ----------
    dataset
        A instance of :class:`~gaitmap_datasets.Kluge2017` or a path to a directory containing the dataset.
    cv_iterator
        A cross-validation iterator or the number of folds to use.
    cv_params
        Additional parameters to pass to the tpcp cross-validation function.

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
    cv_iterator: int | BaseCrossValidator | Iterator | None = field(
        default_factory=lambda: StratifiedGroupKFold(n_splits=5)
    )
    cv_params: dict | None = None

    VERSION = "1.0.0"

    cv_results_: dict = field(init=False)

    def run(self, optimizer: BaseOptimize):
        """Run cross-validation with the prescribed scorer and dataset splits."""
        with self._measure_time():
            self.optimizer = optimizer
            self.dataset_ = self._resolve_dataset()
            cv_params = {} if self.cv_params is None else self.cv_params

            self.cv_results_ = cross_validate(
                optimizable=self.optimizer,
                dataset=self.dataset_,
                cv=DatasetSplitter(self._get_cv_iterator(), groupby=["participant"], stratify=["patient"]),
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
        return final_scorer

    @classmethod
    def get_imu_data(
        cls,
        datapoint: ChallengeDataset,
    ):
        """Select the IMU input required by this challenge."""
        return datapoint.data

    @classmethod
    def get_ground_truth_parameter(cls, datapoint: ChallengeDataset):
        """Calculate reference parameters from the reference system."""
        fake_events = {
            k: ev.assign(min_vel=pd.array([pd.NA] * len(ev), dtype="Int64"))
            for k, ev in datapoint.mocap_events_.items()
        }

        tp = TemporalParameterCalculation(expected_stride_type="ic").calculate(
            stride_event_list=fake_events,
            sampling_rate_hz=datapoint.mocap_sampling_rate_hz_,
        )

        per_stride_trajectory = {k: v["ankle"][GF_POS] for k, v in datapoint.marker_position_per_stride_.items()}

        sp = SpatialParameterCalculation(expected_stride_type="ic").calculate(
            positions=per_stride_trajectory,
            orientations=None,
            stride_event_list=fake_events,
            sampling_rate_hz=datapoint.mocap_sampling_rate_hz_,
        )

        return {k: pd.concat([tp.parameters_[k], sp.parameters_[k]], axis=1) for k in ["left", "right"]}

    @classmethod
    def get_aggregated_ground_truth_parameter(cls, datapoint: ChallengeDataset):
        """Average reference gait parameters across strides and sensors."""
        return pd.concat(cls.get_ground_truth_parameter(datapoint)).mean()
