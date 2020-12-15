"""A helper class for common utilities TrajectoryReconstructionWrapper classes."""
import warnings
from typing import Optional, Tuple, Dict, Sequence, TypeVar, Hashable

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from typing_extensions import Literal

from gaitmap.base import BasePositionMethod, BaseOrientationMethod, BaseTrajectoryMethod
from gaitmap.trajectory_reconstruction.orientation_methods import SimpleGyroIntegration
from gaitmap.trajectory_reconstruction.position_methods import ForwardBackwardIntegration
from gaitmap.utils._algo_helper import invert_result_dictionary, set_params_from_dict
from gaitmap.utils.consts import GF_ORI, GF_VEL, GF_POS, SF_ACC
from gaitmap.utils.datatype_helper import (
    SensorData,
    SingleSensorData,
    set_correct_index,
    get_multi_sensor_names,
    SingleSensorRegionsOfInterestList,
    RegionsOfInterestList,
)
from gaitmap.utils.rotations import rotate_dataset_series, get_gravity_rotation

Self = TypeVar("Self", bound="_TrajectoryReconstructionWrapperMixin")


class _TrajectoryReconstructionWrapperMixin:
    ori_method: Optional[BaseOrientationMethod]
    pos_method: Optional[BasePositionMethod]
    trajectory_method: Optional[BaseTrajectoryMethod]

    data: SensorData
    sampling_rate_hz: float

    _combined_algo_mode: bool
    _integration_regions: RegionsOfInterestList
    _expected_integration_region_index: Sequence[str]

    def __init__(
        self,
        *,
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
            if not (self.ori_method and self.pos_method):
                raise ValueError(
                    "You need to pass either a `ori` and a `pos` method or a single trajectory method. "
                    "You did forget to pass either an `ori` or an `pos` method."
                )
            if isinstance(self.ori_method, BaseTrajectoryMethod) or isinstance(self.pos_method, BaseTrajectoryMethod):
                warnings.warn(
                    "You passed a trajectory method as ori or pos method."
                    "This will still work, but only the orientation or position of the results will be used."
                    "Did you mean to pass it as `trajectory_method`?"
                )
            self._combined_algo_mode = False

    def _estimate(self: Self, dataset_type: Literal["single", "multi"]) -> Self:
        if dataset_type == "single":
            results = self._estimate_single_sensor(self.data, self._integration_regions)
        else:
            results_dict: Dict[Hashable, Dict[str, pd.DataFrame]] = dict()
            for sensor in get_multi_sensor_names(self.data):
                results_dict[sensor] = self._estimate_single_sensor(
                    self.data[sensor], self._integration_regions[sensor]
                )
            results = invert_result_dictionary(results_dict)
        set_params_from_dict(self, results, result_formatting=True)
        return self

    def _estimate_single_sensor(
        self, data: SingleSensorData, integration_regions: SingleSensorRegionsOfInterestList,
    ) -> Dict[str, pd.DataFrame]:
        integration_regions = set_correct_index(integration_regions, self._expected_integration_region_index)
        full_index = tuple((*self._expected_integration_region_index, "sample"))
        orientation = dict()
        velocity = dict()
        position = dict()
        if len(integration_regions) == 0:
            index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=full_index)
            return {
                "orientation": pd.DataFrame(columns=GF_ORI, index=index.copy()),
                "velocity": pd.DataFrame(columns=GF_VEL, index=index.copy()),
                "position": pd.DataFrame(columns=GF_POS, index=index.copy()),
            }
        for r_id, i_region in integration_regions.iterrows():
            i_start, i_end = (int(i_region["start"]), int(i_region["end"]))
            i_orientation, i_velocity, i_position = self._estimate_region(data, i_start, i_end)
            orientation[r_id] = pd.DataFrame(i_orientation.as_quat(), columns=GF_ORI)
            velocity[r_id] = pd.DataFrame(i_velocity, columns=GF_VEL)
            position[r_id] = pd.DataFrame(i_position, columns=GF_POS)
        orientation_df = pd.concat(orientation)
        orientation_df.index = orientation_df.index.rename(full_index)
        velocity_df = pd.concat(velocity)
        velocity_df.index = velocity_df.index.rename(full_index)
        position_df = pd.concat(position)
        position_df.index = position_df.index.rename(full_index)
        return {"orientation": orientation_df, "velocity": velocity_df, "position": position_df}

    def _estimate_region(
        self, data: SingleSensorData, start: int, end: int
    ) -> Tuple[Rotation, pd.DataFrame, pd.DataFrame]:
        stride_data = data.iloc[start:end].copy()
        initial_orientation = self._calculate_initial_orientation(data, start)

        if self._combined_algo_mode is False:
            # For the type-checker
            assert self.ori_method is not None
            assert self.pos_method is not None
            # Apply the orientation method
            ori_method = self.ori_method.clone().set_params(initial_orientation=initial_orientation)
            orientation = ori_method.estimate(stride_data, sampling_rate_hz=self.sampling_rate_hz).orientation_object_

            rotated_stride_data = rotate_dataset_series(stride_data, orientation[:-1])
            # Apply the Position method
            pos_method = self.pos_method.clone().estimate(rotated_stride_data, sampling_rate_hz=self.sampling_rate_hz)
            velocity = pos_method.velocity_
            position = pos_method.position_
        else:
            # For the type-checker
            assert self.trajectory_method is not None
            trajectory_method = self.trajectory_method.clone().set_params(initial_orientation=initial_orientation)
            trajectory_method = trajectory_method.estimate(stride_data, sampling_rate_hz=self.sampling_rate_hz)
            orientation = trajectory_method.orientation_object_
            velocity = trajectory_method.velocity_
            position = trajectory_method.position_
        return orientation, velocity, position

    def _calculate_initial_orientation(self, data: SingleSensorData, start: int) -> Rotation:
        raise NotImplementedError()


def _initial_orientation_from_start(data: SingleSensorData, start: int, align_window_width: int) -> Rotation:
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
        If the value is 0 only the start sample is considered.

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
    acc = (data[SF_ACC].iloc[start_sample : end_sample + 1]).median()
    # get_gravity_rotation assumes [0, 0, 1] as gravity
    return get_gravity_rotation(acc)
