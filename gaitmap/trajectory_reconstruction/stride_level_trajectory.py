"""Wrapper to apply position and orientation estimation to each stride of a dataset."""
import warnings
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from gaitmap.base import (
    BaseOrientationMethod,
    BaseType,
    BasePositionMethod,
    BaseTrajectoryReconstructionWrapper,
    BaseTrajectoryMethod,
)
from gaitmap.trajectory_reconstruction.orientation_methods import SimpleGyroIntegration
from gaitmap.trajectory_reconstruction.position_methods import ForwardBackwardIntegration
from gaitmap.utils.consts import GF_ORI, SF_ACC, GF_VEL, GF_POS, SL_INDEX
from gaitmap.utils.dataset_helper import (
    Dataset,
    StrideList,
    SingleSensorDataset,
    get_multi_sensor_dataset_names,
    set_correct_index,
    is_dataset,
    is_stride_list,
)
from gaitmap.utils.exceptions import ValidationError
from gaitmap.utils.rotations import get_gravity_rotation, rotate_dataset_series


class StrideLevelTrajectory(BaseTrajectoryReconstructionWrapper):
    """Estimate the trajectory over the duration of a stride by considering each stride individually.

    You can select a method for the orientation estimation and a method for the position estimation (or a combined
    method).
    These methods will then be applied to each stride.
    This class will calculate the initial orientation of each stride assuming that it starts at a region of minimal
    movement (`min_vel`).
    Some methods to dedrift the orientation or position will additionally assume that the stride also ends in a
    static period.
    Check the documentation of the individual ori and pos methods for details.

    Attributes
    ----------
    orientation_
        Output of the selected orientation method applied per stride.
        The first orientation is obtained by aligning the acceleration data at the start of each stride with gravity.
        Therefore, +-`half_window_width` around each start of the stride are used as we assume the start of the stride
        to have minimal velocity.
        This contains `len(data) + 1` orientations for each stride, as the initial orientation is included in the
        output.
    position_
        Output of the selected position method applied per stride.
        The initial value of the postion is assumed to be [0, 0, 0].
        This contains `len(data) + 1` values for each stride, as the initial orientation is included in the
        output.
    velocity_
        The velocity as provided by the selected position method applied per stride.
        The initial value of the velocity is assumed to be [0, 0, 0].
        This contains `len(data) + 1` values for each stride, as the initial orientation is included in the
        output.

    Parameters
    ----------
    ori_method
        An instance of any available orientation method with the desired parameters set.
        This method is called with the data of each stride to actually calculate the orientation.
        Note, the the `initial_orientation` parameter of this method will be overwritten, as this class estimates new
        per-stride initial orientations based on the mid-stance assumption.
    pos_method
        An instance of any available position method with the desired parameters set.
        This method is called with the data of each stride to actually calculate the position.
        The provided data is already transformed into the global frame using the orientations calculated by the
        `ori_method` on the same stride.
    trajectory_method
        Instead of providind a separate `ori_method` and `pos_method`, a single `trajectory_method` can be provided
        that calculated the orientation and the position in one go.
        This method is called with the data of each stride.
        If a `trajectory_method` is provided the values for `ori_method` and `pos_method` are ignored.
        Note, the the `initial_orientation` parameter of this method will be overwritten, as this class estimates new
        per-stride initial orientations based on the mid-stance assumption.
    align_window_width
        This is the width of the window that will be used to align the beginning of the signal of each stride with
        gravity. To do so, half the window size before and half the window size after the start of the stride will
        be used to obtain the median value of acceleration data in this phase.
        Note, that +-`np.floor(align_window_size/2)` around the start sample will be used for this. For the first
        stride, start of the stride might coincide with the start of the signal. In that case the start of the window
        would result in a negative index, thus the window to get the initial orientation will be reduced (from 0 to
        `start+np.floor(align_window_size/2)`)

    Other Parameters
    ----------------
    data
        The data passed to the `estimate` method.
    stride_event_list
        The event list passed to the `estimate` method.
    sampling_rate_hz
        The sampling rate of the data.

    Examples
    --------
    You can pick any orientation and any position estimation method that is implemented for this wrapper.

    >>> from gaitmap.trajectory_reconstruction import SimpleGyroIntegration
    >>> from gaitmap.trajectory_reconstruction import ForwardBackwardIntegration
    >>> # Create custom instances of the methods you want to use
    >>> ori_method = SimpleGyroIntegration()
    >>> pos_method = ForwardBackwardIntegration()
    >>> # Create an instance of the wrapper
    >>> per_stride_traj = StrideLevelTrajectory(ori_method=ori_method, pos_method=pos_method)
    >>> # Apply the method
    >>> data = ...
    >>> sampling_rate_hz = 204.8
    >>> stride_list = ...
    >>> per_stride_traj = per_stride_traj.estimate(
    ...                        data,
    ...                        stride_event_list=stride_list,
    ...                        sampling_rate_hz=sampling_rate_hz
    ... )
    >>> per_stride_traj.position_
    <Dataframe or dict with all the positions per stride>
    >>> per_stride_traj.orientation_
    <Dataframe or dict with all the orientations per stride>


    See Also
    --------
    gaitmap.trajectory_reconstruction: Implemented algorithms for orientation and position estimation

    """

    align_window_width: int
    ori_method: Optional[BaseOrientationMethod]
    pos_method: Optional[BasePositionMethod]
    trajectory_method: Optional[BaseTrajectoryMethod]

    data: Dataset
    stride_event_list: StrideList
    sampling_rate_hz: float

    _combined_algo_mode: bool

    def __init__(
        self,
        ori_method: Optional[BaseOrientationMethod] = SimpleGyroIntegration(),
        pos_method: Optional[BasePositionMethod] = ForwardBackwardIntegration(),
        trajectory_method: Optional[BaseTrajectoryMethod] = None,
        align_window_width: int = 8,
    ):
        self.ori_method = ori_method
        self.pos_method = pos_method
        self.trajectory_method = trajectory_method
        # TODO: Make align window with a second value?
        self.align_window_width = align_window_width

    def estimate(self: BaseType, data: Dataset, stride_event_list: StrideList, sampling_rate_hz: float) -> BaseType:
        """Use the initial rotation and the gyroscope signal to estimate the orientation to every time point .

        Parameters
        ----------
        data
            At least must contain 3D-gyroscope data.
        stride_event_list
            List of events for one or multiple sensors.
            For each stride, the orientation and position will be calculated separately.
        sampling_rate_hz
            Sampling rate with which gyroscopic data was recorded.

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.stride_event_list = stride_event_list

        if self.trajectory_method:
            if not isinstance(self.trajectory_method, BasePositionMethod):
                raise ValueError("The provided `trajectory_method` must be a child class of `BaseTrajectoryMethod`.")
            self._combined_algo_mode = True
        else:
            if self.ori_method and not isinstance(self.ori_method, BaseOrientationMethod):
                raise ValueError("The provided `ori_method` must be a child class of `BaseOrientationMethod`.")
            if self.pos_method and not isinstance(self.pos_method, BasePositionMethod):
                raise ValueError("The provided `pos_method` must be a child class of `BasePositionMethod`.")
            self._combined_algo_mode = False

        dataset_type = is_dataset(data, frame="sensor")
        stride_list_type = is_stride_list(stride_event_list, stride_type="min_vel")

        if dataset_type != stride_list_type:
            raise ValidationError(
                "An invalid combination of stride list and dataset was provided."
                "The dataset is {} sensor and the stride list is {} sensor.".format(dataset_type, stride_list_type)
            )
        if dataset_type == "single":
            self.orientation_, self.velocity_, self.position_ = self._estimate_single_sensor(
                self.data, self.stride_event_list
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
            out = self._estimate_single_sensor(self.data[i_sensor], self.stride_event_list[i_sensor])
            orientation[i_sensor] = out[0]
            velocity[i_sensor] = out[1]
            position[i_sensor] = out[2]
        return orientation, velocity, position

    def _estimate_single_sensor(
        self, data: SingleSensorDataset, event_list: StrideList
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        event_list = set_correct_index(event_list, SL_INDEX)
        rotation = dict()
        velocity = dict()
        position = dict()
        if len(event_list) == 0:
            index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=["s_id", "sample"])
            return (
                pd.DataFrame(columns=GF_ORI, index=index.copy()),
                pd.DataFrame(columns=GF_VEL, index=index.copy()),
                pd.DataFrame(columns=GF_POS, index=index.copy()),
            )
        for s_id, i_stride in event_list.iterrows():
            i_start, i_end = (int(i_stride["start"]), int(i_stride["end"]))
            i_rotation, i_velocity, i_position = self._estimate_stride(data, i_start, i_end)
            rotation[s_id] = pd.DataFrame(i_rotation.as_quat(), columns=GF_ORI)
            velocity[s_id] = pd.DataFrame(i_velocity, columns=GF_VEL)
            position[s_id] = pd.DataFrame(i_position, columns=GF_POS)
        rotation = pd.concat(rotation)
        rotation.index = rotation.index.rename(("s_id", "sample"))
        velocity = pd.concat(velocity)
        velocity.index = velocity.index.rename(("s_id", "sample"))
        position = pd.concat(position)
        position.index = position.index.rename(("s_id", "sample"))
        return rotation, velocity, position

    def _estimate_stride(
        self, data: SingleSensorDataset, start: int, end: int
    ) -> Tuple[Rotation, pd.DataFrame, pd.DataFrame]:
        stride_data = data.iloc[start:end].copy()
        initial_orientation = self.calculate_initial_orientation(data, start, self.align_window_width)

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

    @staticmethod
    def calculate_initial_orientation(data: SingleSensorDataset, start: int, align_window_width: float) -> Rotation:
        """Calculate the initial orientation for a section of data using a gravity alignment.

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
