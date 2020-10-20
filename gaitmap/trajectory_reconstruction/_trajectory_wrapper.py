"""A helper class for common utilities TrajectoryReconstructionWrapper classes."""
import warnings
from typing import Optional, Tuple, Dict, Sequence, TypeVar

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from typing_extensions import Literal

from gaitmap.base import BasePositionMethod, BaseOrientationMethod, BaseTrajectoryMethod
from gaitmap.trajectory_reconstruction.orientation_methods import SimpleGyroIntegration
from gaitmap.trajectory_reconstruction.position_methods import ForwardBackwardIntegration
from gaitmap.utils.consts import GF_ORI, GF_VEL, GF_POS, SF_ACC
from gaitmap.utils.dataset_helper import (
    Dataset,
    SingleSensorDataset,
    set_correct_index,
    get_multi_sensor_dataset_names,
)
from gaitmap.utils.rotations import rotate_dataset_series, get_gravity_rotation

RegionType = TypeVar("RegionType")
RegionTypeSingle = TypeVar("RegionTypeSingle")


class _TrajectoryReconstructionWrapperMixin:
    ori_method: Optional[BaseOrientationMethod]
    pos_method: Optional[BasePositionMethod]
    trajectory_method: Optional[BaseTrajectoryMethod]

    data: Dataset
    sampling_rate_hz: float

    _combined_algo_mode: bool
    _integration_regions: RegionType
    _expected_integration_region_index: Sequence[str]

    def __init__(
        self,
        ori_method: Optional[BaseOrientationMethod] = SimpleGyroIntegration(),
        pos_method: Optional[BasePositionMethod] = ForwardBackwardIntegration(),
        trajectory_method: Optional[BaseTrajectoryMethod] = None,
    ):
        self.ori_method = ori_method
        self.pos_method = pos_method
        self.trajectory_method = trajectory_method

    def _validate_methods(self):
        if self.trajectory_method:
            if not isinstance(self.trajectory_method, BaseTrajectoryMethod):
                raise ValueError("The provided `trajectory_method` must be a child class of `BaseTrajectoryMethod`.")
            self._combined_algo_mode = True
        else:
            if self.ori_method and not isinstance(self.ori_method, BaseOrientationMethod):
                raise ValueError("The provided `ori_method` must be a child class of `BaseOrientationMethod`.")
            if self.pos_method and not isinstance(self.pos_method, BasePositionMethod):
                raise ValueError("The provided `pos_method` must be a child class of `BasePositionMethod`.")
            self._combined_algo_mode = False

    def _estimate(self, dataset_type: Literal["single", "multi"]):
        if dataset_type == "single":
            self.orientation_, self.velocity_, self.position_ = self._estimate_single_sensor(
                self.data, self._integration_regions
            )
        else:
            self.orientation_, self.velocity_, self.position_ = self._estimate_multi_sensor()

        return self

    def _estimate_multi_sensor(
        self,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        orientation = dict()
        velocity = dict()
        position = dict()
        for i_sensor in get_multi_sensor_dataset_names(self.data):
            out = self._estimate_single_sensor(self.data[i_sensor], self._integration_regions[i_sensor])
            orientation[i_sensor] = out[0]
            velocity[i_sensor] = out[1]
            position[i_sensor] = out[2]
        return orientation, velocity, position

    def _estimate_single_sensor(
        self, data: SingleSensorDataset, integration_regions: RegionTypeSingle,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        integration_regions = set_correct_index(integration_regions, self._expected_integration_region_index)
        full_index = tuple((*self._expected_integration_region_index, "sample"))
        rotation = dict()
        velocity = dict()
        position = dict()
        if len(integration_regions) == 0:
            index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=full_index)
            return (
                pd.DataFrame(columns=GF_ORI, index=index.copy()),
                pd.DataFrame(columns=GF_VEL, index=index.copy()),
                pd.DataFrame(columns=GF_POS, index=index.copy()),
            )
        for r_id, i_region in integration_regions.iterrows():
            i_start, i_end = (int(i_region["start"]), int(i_region["end"]))
            i_rotation, i_velocity, i_position = self._estimate_region(data, i_start, i_end)
            rotation[r_id] = pd.DataFrame(i_rotation.as_quat(), columns=GF_ORI)
            velocity[r_id] = pd.DataFrame(i_velocity, columns=GF_VEL)
            position[r_id] = pd.DataFrame(i_position, columns=GF_POS)
        rotation = pd.concat(rotation)
        rotation.index = rotation.index.rename(full_index)
        velocity = pd.concat(velocity)
        velocity.index = velocity.index.rename(full_index)
        position = pd.concat(position)
        position.index = position.index.rename(full_index)
        return rotation, velocity, position

    def _estimate_region(
        self, data: SingleSensorDataset, start: int, end: int
    ) -> Tuple[Rotation, pd.DataFrame, pd.DataFrame]:
        stride_data = data.iloc[start:end].copy()
        initial_orientation = self._calculate_initial_orientation(data, start)

        if self._combined_algo_mode is False:
            # Apply the orientation method
            ori_method = self.ori_method.clone().set_params(initial_orientation=initial_orientation)
            orientation = ori_method.estimate(stride_data, sampling_rate_hz=self.sampling_rate_hz).orientation_object_

            rotated_stride_data = rotate_dataset_series(stride_data, orientation[:-1])
            # Apply the Position method
            pos_method = self.pos_method.clone().estimate(rotated_stride_data, sampling_rate_hz=self.sampling_rate_hz)
            velocity = pos_method.velocity_
            position = pos_method.position_
        else:
            trajectory_method = self.trajectory_method.clone().set_params(initial_orientation=initial_orientation)
            trajectory_method = trajectory_method.estimate(stride_data, sampling_rate_hz=self.sampling_rate_hz)
            orientation = trajectory_method.orientation_object_
            velocity = trajectory_method.velocity_
            position = trajectory_method.position_
        return orientation, velocity, position

    def _calculate_initial_orientation(self, data: SingleSensorDataset, start: int) -> Rotation:
        raise NotImplementedError()


def _initial_orientation_from_start(data: SingleSensorDataset, start: int, align_window_width: int) -> Rotation:
    """Calculate the initial orientation for a section of data using a gravity alignment on the first n samples.

    Parameters
    ----------
    data
        The full data of a recording
    start
        The start value in samples of the section of interest in data
    align_window_width
        The size of the window around start that is considered for the alignment.
        The window is centered around start.

    Returns
    -------
    initial_orientation
        The initial orientation, which would align the start of the data section with gravity.

    """
    half_window = int(np.floor(align_window_width / 2))
    if start - half_window >= 0:
        start_sample = start - half_window
    else:
        start_sample = 0
        warnings.warn("Could not use complete window length for initializing orientation.")
    if start + half_window < len(data):
        end_sample = start + half_window
    else:
        end_sample = len(data) - 1
        warnings.warn("Could not use complete window length for initializing orientation.")
    acc = (data[SF_ACC].iloc[start_sample:end_sample]).median()
    # get_gravity_rotation assumes [0, 0, 1] as gravity
    return get_gravity_rotation(acc)
