"""Exercise interchangeable algorithms through the reusable pipeline's run interface."""

from types import SimpleNamespace

import pandas as pd
import pytest

from gaitmap.base import BaseEventDetection, BaseStrideSegmentation, BaseTrajectoryReconstructionWrapper
from gaitmap.parameters import SpatialParameterCalculation, TemporalParameterCalculation
from gaitmap.pipeline import GenericGaitPipeline


def sensor_data():
    return pd.DataFrame({"acc_x": [2.0] * 200, "acc_y": 3.0, "acc_z": 9.81, "gyr_x": 4.0, "gyr_y": 5.0, "gyr_z": 6.0})


class SignalSegmentation(BaseStrideSegmentation):
    def segment(self, data, sampling_rate_hz, **kwargs):
        def segment_one(frame):
            return pd.DataFrame(
                {"start": [0], "end": [100 + int(frame["gyr_pa"].iloc[0])]}, index=pd.Index([0], name="s_id")
            )

        self.stride_list_ = (
            {key: segment_one(frame) for key, frame in data.items()} if isinstance(data, dict) else segment_one(data)
        )
        return self


class FixedEvents(BaseEventDetection):
    def detect(self, data, stride_list, sampling_rate_hz, **kwargs):
        def events_one(strides):
            return strides.assign(min_vel=0, pre_ic=-10, tc=strides["end"] - 50, ic=strides["end"] - 10)

        self.min_vel_event_list_ = (
            {key: events_one(frame) for key, frame in stride_list.items()}
            if isinstance(stride_list, dict)
            else events_one(stride_list)
        )
        return self


class LinearTrajectory(BaseTrajectoryReconstructionWrapper):
    def __init__(self, scale=1.0):
        self.scale = scale

    def estimate(self, data, stride_event_list, *, sampling_rate_hz):
        def trajectory_one(frame, events):
            return pd.DataFrame(
                {"pos_x": [0.0, self.scale * frame["acc_x"].iloc[0]], "pos_y": 0.0, "pos_z": 0.0},
                index=pd.MultiIndex.from_tuples([(0, 0), (0, events["end"].iloc[0])], names=["s_id", "sample"]),
            )

        self.position_ = (
            {key: trajectory_one(frame, stride_event_list[key]) for key, frame in data.items()}
            if isinstance(data, dict)
            else trajectory_one(data, stride_event_list)
        )
        self.orientation_ = None
        return self


def make_pipeline(**kwargs):
    return GenericGaitPipeline(
        stride_segmentation=SignalSegmentation(),
        event_detection=FixedEvents(),
        trajectory_reconstruction=LinearTrajectory(),
        temporal_parameter_calculation=TemporalParameterCalculation(),
        spatial_parameter_calculation=SpatialParameterCalculation(calculate_only=("stride_length",)),
        **kwargs,
    )


def test_runs_on_a_structural_datapoint_and_keeps_algorithm_templates_unmodified():
    # This object has no tpcp or gaitmap base class.
    datapoint = SimpleNamespace(data=sensor_data(), sampling_rate_hz=100.0)
    pipeline = make_pipeline(single_sensor_side="left")
    assert pipeline.safe_run(datapoint) is pipeline
    assert pipeline.parameters_.loc[0, "stride_time"] == pytest.approx(0.96)
    assert pipeline.parameters_.loc[0, "stride_length"] == pytest.approx(2.0)
    assert pipeline.aggregated_gait_parameters_["stride_length"] == pytest.approx(2.0)
    assert not hasattr(pipeline.stride_segmentation, "stride_list_")
    assert not hasattr(pipeline.trajectory_reconstruction, "position_")


def test_left_and_right_sensors_use_their_foot_frames():
    datapoint = SimpleNamespace(
        data={"left_sensor": sensor_data(), "right_sensor": sensor_data()}, sampling_rate_hz=100.0
    )
    pipeline = make_pipeline().safe_run(datapoint)
    assert pipeline.parameters_["left_sensor"].loc[0, "stride_time"] == pytest.approx(0.96)
    assert pipeline.parameters_["right_sensor"].loc[0, "stride_time"] == pytest.approx(1.04)
    assert pipeline.aggregated_gait_parameters_["stride_time"] == pytest.approx(1.0)


def test_nested_algorithm_parameters_change_results_on_a_fresh_run():
    datapoint = SimpleNamespace(data=sensor_data(), sampling_rate_hz=100.0)
    pipeline = make_pipeline(single_sensor_side="right")
    pipeline.safe_run(datapoint)
    pipeline.set_params(trajectory_reconstruction__scale=3.0)
    pipeline.safe_run(datapoint)
    assert pipeline.parameters_.loc[0, "stride_length"] == pytest.approx(6.0)
    assert pipeline.parameters_.loc[0, "stride_time"] == pytest.approx(1.04)


def test_composes_builtin_algorithms_on_real_imu_data(healthy_example_imu_data):
    pytest.importorskip("gaitmap_mad")
    from gaitmap.event_detection import HerzerEventDetection
    from gaitmap.stride_segmentation import BarthDtw

    pipeline = GenericGaitPipeline(stride_segmentation=BarthDtw(), event_detection=HerzerEventDetection())
    pipeline.safe_run(SimpleNamespace(data=healthy_example_imu_data, sampling_rate_hz=204.8))
    assert set(pipeline.parameters_) == {"left_sensor", "right_sensor"}
    assert all(len(parameters) > 10 for parameters in pipeline.parameters_.values())
    assert 0.5 < pipeline.aggregated_gait_parameters_["stride_length"] < 2.0
    assert 0.5 < pipeline.aggregated_gait_parameters_["stride_time"] < 2.0


def test_empty_stride_results_remain_valid_parameter_tables(healthy_example_imu_data):
    pytest.importorskip("gaitmap_mad")
    from gaitmap.event_detection import HerzerEventDetection
    from gaitmap.stride_segmentation import BarthDtw

    pipeline = GenericGaitPipeline(stride_segmentation=BarthDtw(max_cost=0), event_detection=HerzerEventDetection())
    pipeline.safe_run(SimpleNamespace(data=healthy_example_imu_data, sampling_rate_hz=204.8))
    assert all(parameters.empty for parameters in pipeline.parameters_.values())
    assert pipeline.aggregated_gait_parameters_.isna().all()
