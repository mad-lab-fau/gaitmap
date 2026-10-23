"""Composable gait analysis from stride segmentation to gait parameters."""

from typing import Literal, Self

import pandas as pd
from tpcp import CloneFactory, Pipeline

from gaitmap.base import (
    BaseEventDetection,
    BaseSpatialParameterCalculation,
    BaseStrideSegmentation,
    BaseTemporalParameterCalculation,
    BaseTrajectoryReconstructionWrapper,
)
from gaitmap.parameters import SpatialParameterCalculation, TemporalParameterCalculation
from gaitmap.pipeline.base import BaseGaitDataset
from gaitmap.trajectory_reconstruction import StrideLevelTrajectory
from gaitmap.utils.coordinate_conversion import (
    convert_left_foot_to_fbf,
    convert_right_foot_to_fbf,
    convert_to_fbf,
)
from gaitmap.utils.datatype_helper import is_sensor_data


class GenericGaitPipeline(Pipeline):
    """Run interchangeable foot-mounted gait algorithms on one recording.

    Parameters
    ----------
    stride_segmentation
        Algorithm that segments body-frame data and produces ``stride_list_``.
    event_detection
        Algorithm that detects events in body-frame data and produces
        ``min_vel_event_list_``. Configure it to detect all events needed by
        the parameter calculations.
    trajectory_reconstruction
        Algorithm that estimates ``position_`` and ``orientation_`` from the
        sensor-frame data and the detected minimum-velocity strides.
    temporal_parameter_calculation
        Algorithm that calculates temporal parameters from minimum-velocity strides.
    spatial_parameter_calculation
        Algorithm that calculates spatial parameters from these strides and trajectories.
    single_sensor_side
        Foot side for single-sensor input. Required for single-sensor recordings.
    left_like, right_like
        Sensor-name substrings used to select left and right foot sensors in
        multi-sensor input, as in ``convert_to_fbf``. Every input sensor must
        belong to one foot. Defaults match ``left_sensor`` and ``right_sensor``.

    Attributes
    ----------
    stride_list_
        Segmented stride borders before event detection.
    stride_event_list_
        Detected events and minimum-velocity stride borders.
    position_, orientation_
        Per-stride trajectories from the trajectory reconstruction algorithm.
    parameters_
        Temporal and spatial parameters joined by stride ID. A DataFrame for
        single-sensor input, or a dictionary of DataFrames for multiple sensors.
    aggregated_gait_parameters_
        Mean parameters across all strides and sensors, with equal weight per stride.
    stride_segmentation_, event_detection_, trajectory_reconstruction_
        Executed clones of the configured algorithms, including their results.
    temporal_parameter_calculation_, spatial_parameter_calculation_
        Executed parameter calculation algorithms.

    Notes
    -----
    Input must already be aligned to gaitmap's foot sensor frame. Segmentation
    and event detection receive the converted foot body frame; trajectory
    reconstruction receives the original sensor frame. This pipeline does not
    perform sensor alignment or select walking bouts. The parameter calculators
    must use the minimum-velocity stride convention.

    Algorithm constructor arguments remain templates. Each run executes fresh
    clones, and nested parameters can be configured through ``set_params``.
    """

    def __init__(
        self,
        *,
        stride_segmentation: BaseStrideSegmentation,
        event_detection: BaseEventDetection,
        trajectory_reconstruction: BaseTrajectoryReconstructionWrapper = CloneFactory(StrideLevelTrajectory()),
        temporal_parameter_calculation: BaseTemporalParameterCalculation = CloneFactory(TemporalParameterCalculation()),
        spatial_parameter_calculation: BaseSpatialParameterCalculation = CloneFactory(SpatialParameterCalculation()),
        single_sensor_side: Literal["left", "right"] | None = None,
        left_like: str = "left",
        right_like: str = "right",
    ) -> None:
        self.stride_segmentation = stride_segmentation
        self.event_detection = event_detection
        self.trajectory_reconstruction = trajectory_reconstruction
        self.temporal_parameter_calculation = temporal_parameter_calculation
        self.spatial_parameter_calculation = spatial_parameter_calculation
        self.single_sensor_side = single_sensor_side
        self.left_like = left_like
        self.right_like = right_like

    def run(self, datapoint: BaseGaitDataset) -> Self:
        """Analyze one recording satisfying the structural dataset protocol."""
        self.datapoint = datapoint
        data = datapoint.data
        sampling_rate_hz = datapoint.sampling_rate_hz
        dataset_type = is_sensor_data(data, frame="sensor")
        if dataset_type == "single":
            if self.single_sensor_side is None:
                raise ValueError("Set single_sensor_side to 'left' or 'right' for single-sensor recordings.")
            body_data = {"left": convert_left_foot_to_fbf, "right": convert_right_foot_to_fbf}[self.single_sensor_side](
                data
            )
        else:
            body_data = convert_to_fbf(data, left_like=self.left_like, right_like=self.right_like)

        self.stride_segmentation_ = self.stride_segmentation.clone().segment(
            body_data, sampling_rate_hz=sampling_rate_hz
        )
        self.stride_list_ = self.stride_segmentation_.stride_list_
        self.event_detection_ = self.event_detection.clone().detect(
            body_data, stride_list=self.stride_list_, sampling_rate_hz=sampling_rate_hz
        )
        self.stride_event_list_ = self.event_detection_.min_vel_event_list_
        self.trajectory_reconstruction_ = self.trajectory_reconstruction.clone().estimate(
            data, stride_event_list=self.stride_event_list_, sampling_rate_hz=sampling_rate_hz
        )
        self.position_ = self.trajectory_reconstruction_.position_
        self.orientation_ = self.trajectory_reconstruction_.orientation_
        self.temporal_parameter_calculation_ = self.temporal_parameter_calculation.clone().calculate(
            stride_event_list=self.stride_event_list_, sampling_rate_hz=sampling_rate_hz
        )
        self.spatial_parameter_calculation_ = self.spatial_parameter_calculation.clone().calculate(
            stride_event_list=self.stride_event_list_,
            positions=self.position_,
            orientations=self.orientation_,
            sampling_rate_hz=sampling_rate_hz,
        )
        temporal = self.temporal_parameter_calculation_.parameters_
        spatial = self.spatial_parameter_calculation_.parameters_
        if dataset_type == "single":
            self.parameters_ = pd.concat([temporal, spatial], axis=1)
            self.aggregated_gait_parameters_ = self.parameters_.mean()
        else:
            self.parameters_ = {sensor: pd.concat([temporal[sensor], spatial[sensor]], axis=1) for sensor in temporal}
            self.aggregated_gait_parameters_ = pd.concat(self.parameters_).mean()
        return self


__all__ = ["GenericGaitPipeline"]
