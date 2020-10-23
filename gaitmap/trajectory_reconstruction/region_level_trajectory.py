"""Wrapper to apply position and orientation estimation to multiple regions in a dataset."""
from typing import Optional, Tuple, Union

import pandas as pd
from scipy.spatial.transform import Rotation
from typing_extensions import Literal

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
from gaitmap.utils.consts import ROI_ID_COLS, SL_INDEX, TRAJ_TYPE_COLS
from gaitmap.utils.dataset_helper import (
    Dataset,
    SingleSensorDataset,
    is_dataset,
    RegionsOfInterestList,
    is_regions_of_interest_list,
    SingleSensorRegionsOfInterestList,
    get_single_sensor_regions_of_interest_types,
    StrideList,
    is_stride_list,
    VelocityList,
    OrientationList,
    PositionList,
    is_orientation_list,
    is_position_list,
    is_velocity_list,
    get_multi_sensor_dataset_names,
    SingleSensorStrideList,
    SingleSensorPositionList,
    SingleSensorOrientationList,
    SingleSensorVelocityList,
    set_correct_index,
    get_single_sensor_trajectory_list_types,
)
from gaitmap.utils.exceptions import ValidationError


class RegionLevelTrajectory(BaseTrajectoryReconstructionWrapper, _TrajectoryReconstructionWrapperMixin):
    """Estimate the trajectory over the duration of an entire gait sequence or region of interest.

    This class will take any of the implemented orientation, position, and trajectory methods and apply them to all
    regions of interest or gait sequences provided.
    Note that this class assumes that each gait sequence starts with a static resting period and certain methods might
    require the integration regions to end in a static region as well.
    Check the documentation of the individual methods for details.

    Note, that by default this wrapper uses simple integration methods.
    These will not provide the best results under most scenarios, but as they do not have any tunable parameters,
    it is a good start.
    However, it is advisable to chose a more suffisticated method and tune the parameter to your specific usecase.

    Attributes
    ----------
    orientation_
        Output of the selected orientation method applied per region.
        The first orientation is obtained by aligning the acceleration data at the start of each region with gravity.
        This contains `len(data) + 1` orientations for each region, as the initial orientation is included in the
        output.
    position_
        Output of the selected position method applied per region.
        The initial value of the postion is assumed to be [0, 0, 0].
        This contains `len(data) + 1` values for each region, as the initial position is included in the
        output.
    velocity_
        The velocity as provided by the selected position method applied per region.
        The initial value of the velocity is assumed to be [0, 0, 0].
        This contains `len(data) + 1` values for each region, as the initial velocity is included in the
        output.

    Parameters
    ----------
    ori_method
        An instance of any available orientation method with the desired parameters set.
        This method is called with the data of each region to actually calculate the orientation.
        Note, the the `initial_orientation` parameter of this method will be overwritten, as this class estimates new
        per-roi initial orientations.
    pos_method
        An instance of any available position method with the desired parameters set.
        This method is called with the data of each region to actually calculate the position.
        The provided data is already transformed into the global frame using the orientations calculated by the
        `ori_method` on the same stride.
    trajectory_method
        Instead of providing a separate `ori_method` and `pos_method`, a single `trajectory_method` can be provided
        that calculates the orientation and the position in one go.
        This method is called with the data of each region.
        If a `trajectory_method` is provided the values for `ori_method` and `pos_method` are ignored.
        Note, the the `initial_orientation` parameter of this method will be overwritten, as this class estimates new
        per-roi initial orientations.
    align_window_width
        This is the width of the window that will be used to align the beginning of the signal of each region with
        gravity. To do so, half the window size before and half the window size after the start of the region will
        be used to obtain the median value of acceleration data in this phase.
        Note, that +-`np.floor(align_window_size/2)` around the start sample will be used for this. For the first
        region, start of the stride might coincide with the start of the signal. In that case the start of the window
        would result in a negative index, thus the window to get the initial orientation will be reduced (from 0 to
        `start+np.floor(align_window_size/2)`)

    Other Parameters
    ----------------
    data
        The data passed to the `estimate` method.
    regions_of_interest
        The list of regions passed to the `estimate` method.
    sampling_rate_hz
        The sampling rate of the data.

    Examples
    --------
    You can pick any orientation, position, or trajectory estimation method that is implemented for this wrapper.
    However, simple integrations might lead to significant drift if the integration regions are longer.

    >>> from gaitmap.trajectory_reconstruction import RtsKalman
    >>> # Create custom instances of the methods you want to use
    >>> trajectory_method = RtsKalman()
    >>> # Create an instance of the wrapper
    >>> per_region_traj = RegionLevelTrajectory(trajectory_method=trajectory_method)
    >>> # Apply the method
    >>> data = ...
    >>> sampling_rate_hz = 204.8
    >>> roi_list = ...
    >>> per_region_traj = per_region_traj.estimate(
    ...                                            data,
    ...                                            regions_of_interest=roi_list,
    ...                                            sampling_rate_hz=sampling_rate_hz
    ... )
    >>> per_region_traj.position_
    <Dataframe or dict with all the positions per stride>
    >>> per_region_traj.orientation_
    <Dataframe or dict with all the orientations per stride>


    See Also
    --------
    gaitmap.trajectory_reconstruction: Implemented algorithms for orientation and position estimation

    """

    align_window_width: int

    regions_of_interest: RegionsOfInterestList

    def __init__(
        self,
        ori_method: Optional[BaseOrientationMethod] = SimpleGyroIntegration(),
        # TODO: Change default so simple forward integration once this is implemented
        pos_method: Optional[BasePositionMethod] = ForwardBackwardIntegration(),
        trajectory_method: Optional[BaseTrajectoryMethod] = None,
        align_window_width: int = 8,
    ):
        super().__init__(ori_method=ori_method, pos_method=pos_method, trajectory_method=trajectory_method)
        # TODO: Make align window with a second value?
        self.align_window_width = align_window_width

    def estimate(
        self: BaseType, data: Dataset, regions_of_interest: RegionsOfInterestList, sampling_rate_hz: float
    ) -> BaseType:
        """Use the initial rotation and the gyroscope signal to estimate the orientation to every time point .

        Parameters
        ----------
        data
            At least must contain 3D-gyroscope data.
        regions_of_interest
            List of regions for one or multiple sensors.
            For each region, the orientation and position will be calculated separately.
        sampling_rate_hz
            Sampling rate with which gyroscopic data was recorded.

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.regions_of_interest = regions_of_interest
        self._integration_regions = self.regions_of_interest

        self._validate_methods()

        dataset_type = is_dataset(data, frame="sensor")
        roi_list_type = is_regions_of_interest_list(regions_of_interest)

        if dataset_type != roi_list_type:
            raise ValidationError(
                "An invalid combination of ROI list and dataset was provided."
                "The dataset is {} sensor and the ROI list is {} sensor.".format(dataset_type, roi_list_type)
            )

        self._estimate(dataset_type=dataset_type)
        return self

    def estimate_intersect(
        self,
        data: Dataset,
        regions_of_interest: RegionsOfInterestList,
        stride_event_list: StrideList,
        sampling_rate_hz: float,
    ) -> BaseType:
        """
        Notes:
        - This ignores roi of the stride list
        - This cuts out the sample before the start of each stride as initial value (in case stride starts at start
        of region, this will be the estimated initial value)
        -

        Parameters
        ----------
        data
        regions_of_interest
        stride_event_list
        sampling_rate_hz

        Returns
        -------

        """
        roi_list_type = is_regions_of_interest_list(regions_of_interest)
        stride_event_list_type = is_stride_list(stride_event_list)

        if roi_list_type != stride_event_list_type:
            raise ValidationError(
                "An invalid combination of ROI list and stride list was provided."
                "The stride list is {} sensor and the stride list is {} sensor.".format(
                    stride_event_list_type, roi_list_type
                )
            )
        self.estimate(data=data, regions_of_interest=regions_of_interest, sampling_rate_hz=sampling_rate_hz)
        self.orientation_, self.position_, self.velocity_ = self.intersect(
            stride_event_list=stride_event_list, return_data=("orientation", "position", "velocity")
        )
        return self

    def intersect(
        self,
        stride_event_list: StrideList,
        return_data: Tuple[Literal["orientation", "position", "velocity"], ...] = (
            "orientation",
            "position",
            "velocity",
        ),
    ) -> Tuple[Union[PositionList, OrientationList, VelocityList], ...]:
        if self._action_is_applied is False:
            raise ValidationError("You first need to call the `estimate` method before using `intersect`")
        allowed_return_data = ("orientation", "position", "velocity")
        if (
            not isinstance(return_data, (tuple, list))
            or len(return_data) == 0
            or not all(a in ("orientation", "position", "velocity") for a in return_data)
        ):
            raise ValueError("`return_data` must be {} or a subset of it.".format(allowed_return_data))
        validation_methods = {
            "orientation": is_orientation_list,
            "position": is_position_list,
            "velocity": is_velocity_list,
        }
        return_vals = []
        stride_list_type = is_stride_list(stride_event_list)
        for data_name in return_data:
            data = getattr(self, data_name + "_")
            try:
                data_type = validation_methods[data_name](data, "any_roi")
            except ValidationError as e:
                # This should not happen, as we validate result value from our own object.
                # Either, the user has manipulated the values, or called estimate_intersect instead of estimate before.
                # We will check this here.
                try:
                    validation_methods[data_name](data, "stride")
                except ValidationError:
                    raise ValueError(
                        "The existing results do not pass internal validation. "
                        "This can only happen, if you manipulated the outputs of the trajectory before "
                        "calling intersect!"
                    ) from e
                else:
                    raise ValidationError(
                        "The calculated trajectory is already on a per stride level. "
                        "Most likely, you called `estimate_intersect` before. "
                        "In this case there is no need of using intersect again."
                    ) from e
            if data_type != stride_list_type:
                raise ValidationError(
                    "You are trying to intersect the results from a {} sensor dataset with a {} "
                    "sensor stride list".format(data_type, stride_list_type)
                )
            if data_type == "single":
                data = self._intersect(data, self.regions_of_interest, stride_event_list)
            else:
                data = {
                    k: self._intersect(data[k], self.regions_of_interest[k], stride_event_list[k])
                    for k in get_multi_sensor_dataset_names(data)
                }
            return_vals.append(data)
        return tuple(return_vals)

    def _intersect(
        self,
        data: Union[SingleSensorPositionList, SingleSensorOrientationList, SingleSensorVelocityList],
        regions_of_interest: SingleSensorRegionsOfInterestList,
        stride_event_list: SingleSensorStrideList,
    ) -> Union[SingleSensorPositionList, SingleSensorOrientationList, SingleSensorVelocityList]:
        # Note that this ignores the potential roi/gs_id column of the stride list
        stride_event_list = set_correct_index(stride_event_list, SL_INDEX)
        traj_list_type = get_single_sensor_trajectory_list_types(data)
        id_col = TRAJ_TYPE_COLS[traj_list_type]
        regions_of_interest = set_correct_index(regions_of_interest, [id_col])
        data = set_correct_index(data, [id_col, "sample"])
        # TODO: This might be slow, but lets see before we optimize
        #       One way to optimize might be to use search sorted. But this would only work with non overlapping
        #       regions.
        output = {}
        for region_id, region in regions_of_interest.iterrows():
            for s_id, stride in stride_event_list.iterrows():
                if stride["start"] >= region["start"] and stride["end"] <= region["end"]:
                    # This cuts out the n+1 samples for each stride.
                    # The first sample is the value before the stride started.
                    # This is the equivalent to the "initial" position/orientation
                    output[s_id] = (
                        data.loc[region_id]
                        .iloc[int(stride["start"] - region["start"]) : int(stride["end"] - region["start"] + 1)]
                        .reset_index(drop=True)
                    )
        output = pd.concat(output, names=["s_id", "sample"])
        return output

    def _estimate_single_sensor(
        self, data: SingleSensorDataset, integration_regions: SingleSensorRegionsOfInterestList
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Set the class variable to determine the correct index values per dataset.
        self._expected_integration_region_index = [
            ROI_ID_COLS[get_single_sensor_regions_of_interest_types(integration_regions)]
        ]
        return super()._estimate_single_sensor(data, integration_regions)

    def _calculate_initial_orientation(self, data: SingleSensorDataset, start: int) -> Rotation:
        # TODO: Does this way of getting the initial orientation makes sense for longer sequences?
        return _initial_orientation_from_start(data, start, align_window_width=self.align_window_width)
