import warnings
from typing import Dict

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from gaitmap.base import BaseOrientationEstimation, BaseOrientationMethods, BaseType
from gaitmap.trajectory_reconstruction import SimpleGyroIntegration
from gaitmap.utils.consts import GF_ORI, SF_ACC
from gaitmap.utils.dataset_helper import (
    Dataset,
    StrideList,
    is_single_sensor_stride_list,
    is_multi_sensor_stride_list,
    is_single_sensor_dataset,
    is_multi_sensor_dataset,
    SingleSensorDataset,
    get_multi_sensor_dataset_names,
)
from gaitmap.utils.rotations import get_gravity_rotation


class StrideLevelTrajectory(BaseOrientationEstimation):
    """Estimate the trajectory over the duration of a stride by considering each stride individually.

    You can select a method for the orientation estimation and a method for the position estimation.
    These methods will then be applied to each stride.
    This class will calculate the initial orientation of each stride assuming that it starts at a region of minimal
    movement (`min_vel`).
    Further methods to dedrift the orientation or position will additionally assume that the stride also ends in a
    static period.

    Attributes
    ----------
    orientation_
        The first orientation is obtained by aligning the acceleration data at the start of each stride with gravity.
        Therefore, +-`half_window_width` around each start of the stride are used as we assume the start of the stride
        to have minimal velocity.
        The subsequent orientations are obtained from integrating all `len(self.data)` gyroscope samples and therefore
        this attribute contains `len(self.data) + 1` orientations
    position_
    velocity_



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
        Besides the raw data it is provided the orientations calculated by the `ori_method`.
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
        contains gyroscope and acceleration data
    stride_event_list
        A Stride list that will be used to separate `self.data` for integration. For each stride, one sequence of
        orientations will be obtained, all of them result in a Multiindex Pandas Dataframe.
    sampling_rate_hz
        sampling rate of gyroscope data in Hz
    """

    align_window_width: int
    ori_method: BaseOrientationMethods
    pos_method: None

    data: Dataset
    stride_event_list: StrideList
    sampling_rate_hz: float

    def __init__(
        self, ori_method: BaseOrientationMethods = SimpleGyroIntegration(), pos_method=None, align_window_width: int = 8
    ):
        self.ori_method = ori_method
        self.pos_method = pos_method
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
        sampling_rate_hz
            Sampling rate with which gyroscopic data was recorded.

        Notes
        -----
        This function makes use of :py:meth:`~scipy.spatial.transform.Rotation.from_rotvec` to turn the gyro signal of
        each sample into a differential quaternion.
        This means that the rotation between two samples is assumed to be constant around one axis.
        The initial orientation is obtained by aligning acceleration data in the beginning of the signal with gravity.

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.stride_event_list = stride_event_list

        if not is_single_sensor_stride_list(
            stride_event_list, stride_type="min_vel"
        ) and not is_multi_sensor_stride_list(stride_event_list, stride_type="min_vel"):
            raise ValueError("Provided stride event list is not supported by gaitmap")

        if is_single_sensor_dataset(self.data):
            self.estimated_orientations_ = self._estimate_single_sensor(self.data, self.stride_event_list)
        elif is_multi_sensor_dataset(self.data):
            self.estimated_orientations_ = self._estimate_multi_sensor()
        else:
            raise ValueError("Provided data set is not supported by gaitmap")
        return self

    def _estimate_single_sensor(self, data: SingleSensorDataset, event_list: StrideList) -> pd.DataFrame:
        rotations = dict()
        for _, i_stride in event_list.iterrows():
            i_start, i_end = (int(i_stride["start"]), int(i_stride["end"]))
            i_rotations = self._estimate_stride(data, i_start, i_end)
            rotations[i_stride["s_id"]] = pd.DataFrame(i_rotations.as_quat(), columns=GF_ORI)
        rotations = pd.concat(rotations)
        rotations.index = rotations.index.rename(("s_id", "sample"))
        return rotations

    def _estimate_stride(self, data: SingleSensorDataset, start: int, end: int) -> Rotation:
        stride_data = data.iloc[start:end]
        initial_orientation = self._calculate_initial_orientation(data, start)

        # Apply the orientation method
        ori_method = self.ori_method.set_params(initial_orientation=initial_orientation)
        orientation = ori_method.estimate(stride_data, sampling_rate_hz=self.sampling_rate_hz).orientation_

        # Apply the Position method
        if self.pos_method:
            pos_method = self.pos_method.estimate(stride_data, sampling_rate_hz=self.sampling_rate_hz)
            velocity = pos_method.velocity_
            position = pos_method.position_
        return orientation

    def _estimate_multi_sensor(self) -> Dict[str, pd.DataFrame]:
        orientations = dict()
        for i_sensor in get_multi_sensor_dataset_names(self.data):
            orientations[i_sensor] = self._estimate_single_sensor(self.data[i_sensor], self.stride_event_list[i_sensor])
        return orientations

    def _calculate_initial_orientation(self, data: SingleSensorDataset, start) -> Rotation:
        half_window = int(np.floor(self.align_window_width / 2))
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
