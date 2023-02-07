"""Wrapper to apply position and orientation estimation to multiple regions in a dataset."""
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from scipy.spatial.transform import Rotation
from tpcp import CloneFactory, is_action_applied
from typing_extensions import Literal, Self

from gaitmap.base import (
    BaseOrientationMethod,
    BasePositionMethod,
    BaseTrajectoryMethod,
    BaseTrajectoryReconstructionWrapper,
)
from gaitmap.trajectory_reconstruction._trajectory_wrapper import (
    _initial_orientation_from_start,
    _TrajectoryReconstructionWrapperMixin,
)
from gaitmap.trajectory_reconstruction.orientation_methods import SimpleGyroIntegration
from gaitmap.trajectory_reconstruction.position_methods import ForwardBackwardIntegration
from gaitmap.utils.consts import ROI_ID_COLS, SL_INDEX, TRAJ_TYPE_COLS
from gaitmap.utils.datatype_helper import (
    OrientationList,
    PositionList,
    RegionsOfInterestList,
    SensorData,
    SingleSensorData,
    SingleSensorOrientationList,
    SingleSensorPositionList,
    SingleSensorRegionsOfInterestList,
    SingleSensorStrideList,
    SingleSensorVelocityList,
    StrideList,
    VelocityList,
    get_multi_sensor_names,
    get_single_sensor_regions_of_interest_types,
    get_single_sensor_trajectory_list_types,
    is_orientation_list,
    is_position_list,
    is_regions_of_interest_list,
    is_sensor_data,
    is_stride_list,
    is_velocity_list,
    set_correct_index,
)
from gaitmap.utils.exceptions import ValidationError
from gaitmap.utils.stride_list_conversion import intersect_stride_list


class RegionLevelTrajectory(_TrajectoryReconstructionWrapperMixin, BaseTrajectoryReconstructionWrapper):
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
        In case `estimate_intersect` is used, this will not be per region, but per stride.
        Note, that the initial values are not reset per stride.
    position_
        Output of the selected position method applied per region.
        The initial value of each region is assumed to be [0, 0, 0].
        This contains `len(data) + 1` values for each region, as the initial position is included in the
        output.
        In case `estimate_intersect` is used, this will not be per region, but per stride.
        Note, that the initial values are not reset per stride.
    velocity_
        The velocity as provided by the selected position method applied per region.
        The initial value of the velocity is assumed to be [0, 0, 0].
        This contains `len(data) + 1` values for each region, as the initial velocity is included in the
        output.
        In case `estimate_intersect` is used, this will not be per region, but per stride.
        Note, that the initial values are not reset per stride.

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
        Note, the `initial_orientation` parameter of this method will be overwritten, as this class estimates new
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
        The data passed to the `estimate` or `estimate_intersect` method.
    regions_of_interest
        The list of regions passed to the `estimate` or `estimate_intersect` method.
    stride_event_list
        The list of strides passed to the `estimate_intersect`
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
    <Dataframe or dict with all the positions per region>
    >>> per_region_traj.orientation_
    <Dataframe or dict with all the orientations per region>

    If you want to have the trajectory per stride but still use integration methods that could benefit from
    performing the integration on entire regions, you can use `estimate_intersect` to cut out the per-stride
    trajectories.

    >>> from gaitmap.trajectory_reconstruction import SimpleGyroIntegration
    >>> from gaitmap.trajectory_reconstruction import ForwardBackwardIntegration
    >>> # Create custom instances of the methods you want to use
    >>> trajectory_method = RtsKalman()
    >>> # Create an instance of the wrapper
    >>> per_region_traj = RegionLevelTrajectory(trajectory_method=trajectory_method)
    >>> # Apply the method
    >>> data = ...
    >>> sampling_rate_hz = 204.8
    >>> roi_list = ...
    >>> stride_list = ...
    >>> per_region_traj = per_region_traj.estimate_intersect(
    ...                                                      data,
    ...                                                      regions_of_interest=roi_list,
    ...                                                      stride_event_list=stride_list,
    ...                                                      sampling_rate_hz=sampling_rate_hz
    ... )
    >>> per_region_traj.position_
    <Dataframe or dict with all the positions per stride>
    >>> per_region_traj.orientation_
    <Dataframe or dict with all the orientations per stride>


    See Also
    --------
    gaitmap.trajectory_reconstruction: Implemented algorithms for orientation and position estimation

    """

    _action_methods = ("estimate", "estimate_intersect")

    align_window_width: int

    regions_of_interest: RegionsOfInterestList
    stride_event_list: StrideList

    def __init__(
        self,
        *,
        ori_method: Optional[BaseOrientationMethod] = CloneFactory(SimpleGyroIntegration()),
        # TODO: Change default so simple forward integration once this is implemented
        pos_method: Optional[BasePositionMethod] = CloneFactory(ForwardBackwardIntegration()),
        trajectory_method: Optional[BaseTrajectoryMethod] = None,
        align_window_width: int = 8,
    ):
        # TODO: Make align window with a second value?
        self.align_window_width = align_window_width
        super().__init__(ori_method=ori_method, pos_method=pos_method, trajectory_method=trajectory_method)

    def estimate(
        self,
        data: SensorData,
        regions_of_interest: RegionsOfInterestList,
        *,
        stride_event_list: Optional[StrideList] = None,
        sampling_rate_hz: float,
    ) -> Self:
        """Use the initial rotation and the gyroscope signal to estimate the orientation to every time point .

        Parameters
        ----------
        data
            At least must contain 3D-gyroscope and 3D-accelerometer data.
        regions_of_interest
            List of regions for one or multiple sensors.
            For each region, the orientation and position will be calculated separately.
        stride_event_list
            An optional stride event list that will be forwarded to the traj/ori/pos methods.
            The stride event list will be intersected with the region of interest list and the respective pieces will be
            provided to the detect method of the underlying methods.
        sampling_rate_hz
            Sampling rate with which IMU data was recorded.

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        self.regions_of_interest = regions_of_interest
        self.stride_event_list = stride_event_list

        self._validate_methods()

        dataset_type = is_sensor_data(data, frame="sensor")
        roi_list_type = is_regions_of_interest_list(regions_of_interest)

        stride_list_list = None
        if stride_event_list is not None:
            stride_event_list_type = is_stride_list(stride_event_list)

            if roi_list_type != stride_event_list_type:
                raise ValidationError(
                    "An invalid combination of ROI list and stride list was provided. "
                    f"The stride list is {stride_event_list_type} sensor and the ROI list is {roi_list_type} sensor."
                )

            if stride_event_list_type == "single":
                stride_list_list = intersect_stride_list(self.stride_event_list, self.regions_of_interest)
            else:
                stride_list_list = {
                    sensor: intersect_stride_list(stride_list, self.regions_of_interest[sensor])
                    for sensor, stride_list in self.stride_event_list.items()
                }

        if dataset_type != roi_list_type:
            raise ValidationError(
                "An invalid combination of ROI list and dataset was provided. "
                f"The dataset is {dataset_type} sensor and the ROI list is {roi_list_type} sensor."
            )

        self._estimate(
            data=data,
            integration_regions=regions_of_interest,
            dataset_type=dataset_type,
            stride_list_list=stride_list_list,
        )
        return self

    def estimate_intersect(
        self,
        data: SensorData,
        regions_of_interest: RegionsOfInterestList,
        *,
        stride_event_list: StrideList,
        sampling_rate_hz: float,
    ) -> Self:
        """Estimate the trajectory for all regions and then cut out the trajectory of individual strides.

        This basically calls `estimate` and then `intersect` and stores the results in the object.
        This means that the results (`position_`, ...) will be on stride level and not on region level.
        Calculated values outside of strides will simply be dropped.

        See the Notes sections for some caveats.

        Parameters
        ----------
        data
            At least must contain 3D-gyroscope and 3D-accelerometer data.
        regions_of_interest
            List of regions for one or multiple sensors.
            For each region, the orientation and position will be calculated separately.
        stride_event_list
            List of strides within the provided regions.
            The trajectories of these strides will be cut out of the trajectories calculated for the respective region.
        sampling_rate_hz
            Sampling rate with which IMU data was recorded.

        Notes
        -----
        Make sure you follow the these tips to avoid errors:

        - The stride list must have information for all the sensors that are also in the roi list
        - The method cuts out the sample before the start of each stride as initial value (in case stride starts at
          start of region, this will be the estimated initial value).
          This makes the output structure equivalent to `StrideLevelTrajectory`, however, the initial values are not
          reset per stride.
          This means, that for example the position of each stride will not start at [0, 0, 0]
        - If you provide ROI lists that have overlaps (THIS SHOULD BE AVOIDED!) the stride trajectory from the last ROI
          is used.
        - Strides are considered to be part of an ROI, if they are fully contained (start >= roi_start, end <= roi_end).
        - Strides that have only partial or now overlap with a ROI are omitted from the output.

        """
        self.stride_event_list = stride_event_list

        self.estimate(
            data=data,
            regions_of_interest=regions_of_interest,
            stride_event_list=stride_event_list,
            sampling_rate_hz=sampling_rate_hz,
        )
        # pylint: disable=unbalanced-tuple-unpacking
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
        """Cut out the trajectory of individual strides from the region trajectories.

        This method can only be used after `estimate` was called.
        Calling `estimate_intersect` before, does **not** work!

        For more information about how the intersect works, see `estimate_intersect`.

        Parameters
        ----------
        stride_event_list
            List of strides within the provided regions.
            The trajectories of these strides will be cut out of the trajectories calculated for the respective region.
        return_data
            Which results should be returned per stride

        Returns
        -------
        result_tuple
            A tuple with the outputs, following the order provided in `return_data`.

        """
        if is_action_applied(self) is False:
            raise ValidationError("You first need to call the `estimate` method before using `intersect`")
        allowed_return_data = ("orientation", "position", "velocity")
        if (
            not isinstance(return_data, (tuple, list))
            or len(return_data) == 0
            or not all(a in ("orientation", "position", "velocity") for a in return_data)
        ):
            raise ValueError(f"`return_data` must be {allowed_return_data} or a subset of it.")
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
                        "In this case there is no need to use intersect again."
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
                    for k in get_multi_sensor_names(data)
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

        stride_lists = intersect_stride_list(stride_event_list, regions_of_interest)
        output = {}
        for (region_id, _), stride_list in zip(regions_of_interest.iterrows(), stride_lists):
            for s_id, stride in stride_list.iterrows():
                # This cuts out the n+1 samples for each stride.
                # The first sample is the value before the stride started.
                # This is the equivalent to the "initial" position/orientation
                output[s_id] = (
                    data.loc[region_id].iloc[int(stride["start"]) : int(stride["end"] + 1)].reset_index(drop=True)
                )
        output = pd.concat(output, names=["s_id", "sample"])
        return output

    def _estimate_single_sensor(
        self,
        data: SingleSensorData,
        integration_regions: SingleSensorRegionsOfInterestList,
        stride_list_list: Optional[List[SingleSensorStrideList]],
    ) -> Dict[str, pd.DataFrame]:
        # Set the class variable to determine the correct index values per dataset.
        self._expected_integration_region_index = [
            ROI_ID_COLS[get_single_sensor_regions_of_interest_types(integration_regions)]
        ]
        return super()._estimate_single_sensor(data, integration_regions, stride_list_list)

    def _calculate_initial_orientation(self, data: SingleSensorData, start: int) -> Rotation:
        # TODO: Does this way of getting the initial orientation makes sense for longer sequences?
        return _initial_orientation_from_start(data, start, align_window_width=self.align_window_width)
