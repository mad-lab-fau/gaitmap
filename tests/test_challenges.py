"""Run challenge splits and reference calculations without downloading datasets."""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from tpcp import Pipeline
from tpcp.optimize import DummyOptimize

pytest.importorskip("gaitmap_datasets")

from gaitmap.pipeline.challenges.kluge_2017 import Challenge as KlugeChallenge

CHALLENGES = [
    ("stride_segmentation", "EgaitSegmentationValidation2014Challenge", False),
    ("stride_segmentation", "EgaitSegmentationValidation2014OriginalLabelChallenge", False),
    ("stride_segmentation", "SensorPositionComparisonInstepChallenge", False),
    ("parameters", "EgaitAdidas2014Challenge", True),
    ("parameters", "EgaitParameterValidation2013Challenge", False),
    ("parameters", "SensorPositionComparisonInstepChallenge", True),
    ("pipeline", "Kluge2017Challenge", True),
    ("pipeline", "SensorPositionComparisonInstepChallenge", True),
]


class ConstantPipeline(Pipeline):
    def run(self, datapoint):
        self.score_ = 1.0
        return self


def constant_score(pipeline, datapoint):
    return pipeline.safe_run(datapoint).score_


@pytest.mark.parametrize("fold_generator", [False, True])
@pytest.mark.parametrize("family,name,grouped", CHALLENGES)
def test_challenge_runs_and_keeps_participants_in_one_fold(family, name, grouped, fold_generator):
    challenge_class = getattr(import_module(f"gaitmap.{family}.challenges"), name)
    dataset_type = import_module(challenge_class.__module__).ChallengeDataset
    index = pd.DataFrame(
        {
            "participant": [f"p{i}" for i in range(4) for _ in range(2)],
            "test": ["walk"] * 8,
            "cohort": ["healthy"] * 8,
            "patient": ["healthy"] * 8,
            "repetition": [0, 1] * 4,
        }
    )
    dataset = dataset_type(data_folder=Path("/unused"), subset_index=index)

    class CustomScoringChallenge(challenge_class):
        @classmethod
        def get_scorer(cls):
            return constant_score

    cv = (fold for fold in [([0, 1, 2, 3], [4, 5, 6, 7]), ([4, 5, 6, 7], [0, 1, 2, 3])]) if fold_generator else 2
    challenge = CustomScoringChallenge(
        dataset=dataset, cv_iterator=cv, cv_params={"progress_bar": False, "return_train_score": True}
    )
    assert challenge.run(DummyOptimize(ConstantPipeline())) is challenge
    assert list(challenge.cv_results_["test__agg__score"]) == [1.0, 1.0]
    if grouped:
        for train, test in zip(challenge.cv_results_["train__data_labels"], challenge.cv_results_["test__data_labels"]):
            assert {label.participant for label in train}.isdisjoint(label.participant for label in test)


def test_kluge_reference_joins_temporal_and_spatial_parameters_per_stride():
    events = pd.DataFrame(
        {"start": [0, 100], "end": [100, 200], "ic": [0, 100], "tc": [60, 160]},
        index=pd.Index([0, 1], name="s_id"),
    )
    position = pd.DataFrame(
        {"pos_x": [0.0, 1.0, 0.0, 1.0], "pos_y": 0.0, "pos_z": 0.0},
        index=pd.MultiIndex.from_product([[0, 1], [0, 100]], names=["s_id", "sample"]),
    )
    datapoint = SimpleNamespace(
        mocap_events_={foot: events for foot in ["left", "right"]},
        mocap_sampling_rate_hz_=100,
        marker_position_per_stride_={foot: {"ankle": position} for foot in ["left", "right"]},
    )
    reference = KlugeChallenge.get_aggregated_ground_truth_parameter(datapoint)
    assert reference["stride_length"] == pytest.approx(1.0)
    assert reference["stride_time"] == pytest.approx(1.0)
    assert reference["swing_time"] == pytest.approx(0.4)


class NamedParameterPipeline(Pipeline):
    def run(self, datapoint):
        self.aggregated_gait_parameters_ = pd.Series({"stride_length": 1.2}, name="mean")
        return self


@pytest.mark.parametrize("source", ["kluge_2017", "sensor_position_comparison_instep"])
def test_full_pipeline_scorers_accept_named_parameter_series(source):
    module = import_module(f"gaitmap.pipeline.challenges.{source}")
    events = pd.DataFrame(
        {"start": [0, 100], "end": [100, 200], "ic": [80, 180], "tc": [40, 140]},
        index=pd.Index([0, 1], name="s_id"),
    )
    position = pd.DataFrame(
        {"pos_x": [0.0, 1.0, 0.0, 1.0], "pos_y": 0.0, "pos_z": 0.0},
        index=pd.MultiIndex.from_product([[0, 1], [0, 100]], names=["s_id", "sample"]),
    )
    if source == "kluge_2017":
        events["ic"] = events["start"]
        events["tc"] = events["start"] + 60
        markers = {foot: {"ankle": position} for foot in ["left", "right"]}
    else:
        events["min_vel"] = events["start"] + 90
        xyz = position.rename(columns=lambda col: col.removeprefix("pos_"))
        markers = {foot: pd.concat({f"{foot[0]}_fcc": xyz}, axis=1) for foot in ["left", "right"]}
    datapoint = SimpleNamespace(
        mocap_events_={foot: events for foot in ["left", "right"]},
        mocap_sampling_rate_hz_=100,
        marker_position_per_stride_=markers,
    )
    scores = module.final_scorer(NamedParameterPipeline(), datapoint)
    assert scores["stride_length__error"].get_value() == pytest.approx(0.2)


class OriginalLabelPipeline(Pipeline):
    def run(self, datapoint):
        self.stride_list_ = datapoint.segmented_stride_list_original_
        return self


def test_original_label_exported_scorer_uses_original_annotations():
    module = import_module("gaitmap.stride_segmentation.challenges.egait_segmentation_validation_2014_original_label")
    original = pd.DataFrame({"start": [0], "end": [100]}, index=pd.Index([0], name="s_id"))
    new = original + 200
    datapoint = SimpleNamespace(
        sampling_rate_hz=100,
        segmented_stride_list_original_={"left_sensor": original},
        segmented_stride_list_={"left_sensor": new},
    )
    scores = module.final_scorer(OriginalLabelPipeline(), datapoint)
    assert scores["f1_score"] == pytest.approx(1.0)
