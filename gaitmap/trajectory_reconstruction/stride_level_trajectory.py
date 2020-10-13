"""Wrapper to apply position and orientation estimation to each stride of a dataset."""
from typing import Optional

from scipy.spatial.transform import Rotation

from gaitmap.base import (
    BaseOrientationMethod,
    BaseType,
    BasePositionMethod,
    BaseTrajectoryReconstructionWrapper,
    BaseTrajectoryMethod,
)
from gaitmap.trajectory_reconstruction._trajectory_wrapper import (
    _TrajectoryReconstructionWrapperMixin,
    _initial_orientation_from_start,
)
from gaitmap.trajectory_reconstruction.orientation_methods import SimpleGyroIntegration
from gaitmap.trajectory_reconstruction.position_methods import ForwardBackwardIntegration
from gaitmap.utils.consts import SL_INDEX
from gaitmap.utils.dataset_helper import (
    Dataset,
    StrideList,
    SingleSensorDataset,
    is_dataset,
    is_stride_list,
)
from gaitmap.utils.exceptions import ValidationError


class StrideLevelTrajectory(BaseTrajectoryReconstructionWrapper, _TrajectoryReconstructionWrapperMixin):
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
        This contains `len(data) + 1` orientations for each stride, as the initial orientation is included in the
        output.
    position_
        Output of the selected position method applied per stride.
        The initial value of the postion is assumed to be [0, 0, 0].
        This contains `len(data) + 1` values for each stride, as the initial position is included in the
        output.
    velocity_
        The velocity as provided by the selected position method applied per stride.
        The initial value of the velocity is assumed to be [0, 0, 0].
        This contains `len(data) + 1` values for each stride, as the initial velocity is included in the
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

    stride_event_list: StrideList

    _expected_integration_region_index = SL_INDEX

    def __init__(
        self,
        ori_method: Optional[BaseOrientationMethod] = SimpleGyroIntegration(),
        pos_method: Optional[BasePositionMethod] = ForwardBackwardIntegration(),
        trajectory_method: Optional[BaseTrajectoryMethod] = None,
        align_window_width: int = 8,
    ):
        super().__init__(ori_method=ori_method, pos_method=pos_method, trajectory_method=trajectory_method)
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
        self._integration_regions = self.stride_event_list

        self._validate_methods()

        dataset_type = is_dataset(data, frame="sensor")
        stride_list_type = is_stride_list(stride_event_list, stride_type="min_vel")

        if dataset_type != stride_list_type:
            raise ValidationError(
                "An invalid combination of stride list and dataset was provided."
                "The dataset is {} sensor and the stride list is {} sensor.".format(dataset_type, stride_list_type)
            )

        self._estimate(dataset_type=dataset_type)
        return self

    def _calculate_initial_orientation(self, data: SingleSensorDataset, start: int) -> Rotation:
        return _initial_orientation_from_start(data, start, align_window_width=self.align_window_width)
