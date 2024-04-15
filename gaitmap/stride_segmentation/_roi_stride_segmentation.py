"""Wrapper class to apply a stride segmentation to multiple regions of interest in a dataset."""

from copy import deepcopy
from typing import Dict, Generic, Optional, TypeVar, Union

import pandas as pd
from tpcp import get_action_method
from typing_extensions import Literal

from gaitmap.base import BaseStrideSegmentation
from gaitmap.utils._algo_helper import invert_result_dictionary, set_params_from_dict
from gaitmap.utils._types import _Hashable
from gaitmap.utils.consts import ROI_ID_COLS
from gaitmap.utils.datatype_helper import (
    RegionsOfInterestList,
    SensorData,
    SingleSensorRegionsOfInterestList,
    StrideList,
    get_multi_sensor_names,
    get_single_sensor_regions_of_interest_types,
    is_regions_of_interest_list,
    is_sensor_data,
)
from gaitmap.utils.exceptions import ValidationError

StrideSegmentationAlgorithm = TypeVar("StrideSegmentationAlgorithm", bound=BaseStrideSegmentation)

Self = TypeVar("Self", bound="RoiStrideSegmentation")


class RoiStrideSegmentation(BaseStrideSegmentation, Generic[StrideSegmentationAlgorithm]):
    """Apply any stride segmentation algorithms to specific regions of interest in a longer dataset.

    In many cases it is preferable to not apply a stride segmentation algorithm to an entire dataset, but rather
    preselect regions of interest.
    These regions could be defined by some activity recognition algorithms or manually using further knowledge about
    the kind of recording.

    This class allows you to easily loop over all specified regions of interest and call the selected stride
    segmentation algorithm just with the selected piece of data.
    The final stride list is than again concatenated over all regions of interest.

    The class supports different types of input data combinations:

    Single-sensor datasets with single-sensor regions of interest
        If a simple single-sensor dataset and a single-sensor region of interest is supplied, the segmentation method
        will be simply called for each region of interest.
    Synchronised multi-sensor dataset with single-sensor regions of interest
        If a synchronised multi-sensor dataset (multi-level pandas dataframe) is passed and just a single-sensor
        regions of interest list, this list will be applied to all sensors.
        This class will handle looping the rois and looping the individual sensors will be handled by the actual
        segmentation algorithm.
    Multi-sensor dataset with multi-sensor regions of interest
        If a multi-sensor regions of interest list is provided the entries for each sensor will be applied to
        respective datastreams of the dataset.
        In this case this class handles looping over the sensors and over the ROIs.
        The actual segmentation method will be called for each combination of sensor and ROI individually.
        All outputs will become nested dictionaries with the sensor-name at the top level.
        Note that sensors that do not have ROIs specified will not be processed.

    For more information about the valid formats for the regions of interest list, review the
    :ref:`datatype guide <datatypes>`.

    Parameters
    ----------
    segmentation_algorithm
        An instance of a valid segmentation algorithm with all the wanted parameters set.
    s_id_naming
        Controls how the stride ids of the final stride lists are created to ensure they are unique.
        In case of "replace" the stride ids created by the stride segmentation algorithms per ROI are removed and
        replaced with an increasing numerical id.
        In case of "prefix" the original ids are kept and prefixed with "{roi_id}_".
    action_method
        Controls which action method of the wrapped algorithm should be called.
        This is only relevant, if the wrapped algorithm offers mutliple action methods.
        By default the primary action method is used.

    Attributes
    ----------
    stride_list_ : pd.DataFrame or Dictionary of such values
        The final stride list marking the start and the end of each detected stride.
        It further contains a column "gsd_id" or "roi_id" (depending on the roi list used in the input) that
        indicates in which of the ROIs each stride was detected.
        The "start" and "end" values are relative to the start of the dataset (and not the individual ROIs).
        In case a multi-sensor ROI list was used, this will be a dictionary with one stride list per sensor.
        Further information about the outputs can be found in the documentation of the used segmentation algorithm.
    instances_per_roi_ : (Nested) Dictionary of StrideSegmentation algorithm instances
        The actual instances of the stride segmentation algorithm for each ROI.
        They can be used to inspect further results and extract debug information.
        For available values review the documentation of the used stride segmentation algorithm.
        Remember that all values and data in these individual instances are relative to the start of each individual
        ROI.
        In case a multi-sensor ROI list was used, this will be a nested dictionary, with the sensor name as the top
        level and the ROI-ID as the second level.

    Other Parameters
    ----------------
    data
        The data passed to the `segment` method.
    sampling_rate_hz
        The sampling rate of the data
    regions_of_interest
        The regions of interest list defining the start and the end of each region

    Examples
    --------
    >>> # We need our normal raw data (note that the required format might depend on the used segmentation method) and
    ... # And a valid region of interest list
    >>> data = pd.DataFrame(...)
    >>> roi_list = pd.DataFrame(...)
    >>> # Create an instance of your stride segmentation algorithm
    >>> from gaitmap.stride_segmentation import BarthDtw
    >>> dtw = BarthDtw()
    >>> # Now we can use the RoiStrideSegmentation to apply BarthDtw to each ROI
    >>> roi_seg = RoiStrideSegmentation(segmentation_algorithm=dtw)
    >>> roi_seg.segment(data, sampling_rate_hz=100, regions_of_interest=roi_list)
    >>> # Inspect the results
    >>> roi_seg.stride_list_
    # TODO: Add example output
    >>> roi_seg.instances_per_roi_
    {"roi_1": <BarthDtw...>, "roi_2": <BarthDtw ...>, ...}

    """

    segmentation_algorithm: Optional[StrideSegmentationAlgorithm]
    s_id_naming: Literal["replace", "prefix"]
    action_method: Optional[str]

    instances_per_roi_: Union[
        Dict[_Hashable, StrideSegmentationAlgorithm], Dict[_Hashable, Dict[_Hashable, StrideSegmentationAlgorithm]]
    ]
    stride_list_: StrideList

    data: SensorData
    sampling_rate_hz: float
    regions_of_interest: RegionsOfInterestList

    _multi_dataset: bool
    _multi_roi: bool

    def __init__(
        self,
        segmentation_algorithm: Optional[StrideSegmentationAlgorithm] = None,
        s_id_naming: Literal["replace", "prefix"] = "replace",
        action_method: Optional[str] = None,
    ) -> None:
        self.segmentation_algorithm = segmentation_algorithm
        self.s_id_naming = s_id_naming
        self.action_method = action_method

    def segment(
        self: Self,
        data: SensorData,
        sampling_rate_hz: float,
        *,
        regions_of_interest: Optional[RegionsOfInterestList] = None,
        **kwargs,
    ) -> Self:
        """Run the segmentation on each region of interest.

        Parameters
        ----------
        data : array, single-sensor dataframe, or multi-sensor dataset
            The input data.
            For details on the required datatypes review the class docstring.
        sampling_rate_hz
            The sampling rate of the data signal.
        regions_of_interest : single or multi-sensor regions of interest list
            The regions of interest that should be used.
            The segmentation algorithm will be applied to each region individually
        kwargs
            All keyword arguments will be passed to the segment method of the selected `segmentation_algorithm`

        """
        self.regions_of_interest = regions_of_interest
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        self._validate_parameters()
        self._validate_other_parameters()
        # For the type-checker
        assert self.regions_of_interest is not None
        assert self.segmentation_algorithm is not None

        if self._multi_roi:
            # Apply the segmentation to a single dataset in case a multi - sensor roi list is provided
            results_dict: Dict[
                _Hashable, Dict[str, Union[pd.DataFrame, Dict[_Hashable, StrideSegmentationAlgorithm]]]
            ] = {}
            for sensor, roi in self.regions_of_interest.items():
                sensor_data = self.data[sensor]
                results_dict[sensor] = self._segment_single_sensor(
                    self.segmentation_algorithm, sensor_data, self.sampling_rate_hz, roi, sensor_name=sensor, **kwargs
                )
            results = invert_result_dictionary(results_dict)
        else:
            results = self._segment_single_sensor(
                self.segmentation_algorithm, self.data, self.sampling_rate_hz, self.regions_of_interest, **kwargs
            )
        set_params_from_dict(self, results, result_formatting=True)

        return self

    def _segment_single_sensor(
        self,
        segmentation_algorithm: StrideSegmentationAlgorithm,
        sensor_data: pd.DataFrame,
        sampling_rate_hz: float,
        rois: SingleSensorRegionsOfInterestList,
        sensor_name: Optional[_Hashable] = None,
        **kwargs,
    ) -> Dict[str, Union[pd.DataFrame, Dict[_Hashable, StrideSegmentationAlgorithm]]]:
        """Call the the segmentation algorithm for each region of interest and store the instance."""
        rois = rois.reset_index()
        index_col = ROI_ID_COLS[get_single_sensor_regions_of_interest_types(rois)]

        instances_per_roi = {}
        combined_stride_list = {}
        for _, (index, start, end) in rois[[index_col, "start", "end"]].iterrows():
            per_roi_algo = segmentation_algorithm.clone()
            roi_data = sensor_data.iloc[start:end]
            if sensor_name:
                # In case the original dataset was a dict, sensor_name is passed by the parent.
                # Here we ensure that the data passed into segment method is still a dict to avoid potential issues with
                # specific segmentation methods.
                # We basically pass it as a multi-sensor dataset with just a single sensor
                roi_data = {sensor_name: roi_data}
            per_roi_algo = get_action_method(per_roi_algo, self.action_method)(
                data=roi_data, sampling_rate_hz=sampling_rate_hz, **kwargs
            )
            instances_per_roi[index] = per_roi_algo
            stride_list = deepcopy(per_roi_algo.stride_list_)
            if isinstance(stride_list, dict):
                for strides in stride_list.values():
                    # This is an inplace modification!
                    strides[["start", "end"]] += start
            else:
                stride_list[["start", "end"]] += start
            if sensor_name:
                # The reverse of what we did above.
                # For this method a single sensor dataset should be a real single sensor dataset
                combined_stride_list[index] = stride_list[sensor_name]
            else:
                combined_stride_list[index] = stride_list

        combined_stride_list = self._merge_stride_lists(combined_stride_list, index_col)
        return {"stride_list": combined_stride_list, "instances_per_roi": instances_per_roi}

    def _merge_stride_lists(self, stride_lists: Dict[_Hashable, StrideList], index_name: str) -> StrideList:
        """Merge either single or multisensor stride lists.

        The roi id (the dict key in the input), will be a column in the final output dataframe.
        """
        # We assume all algorithms follow convention and return the correct format
        if self._multi_dataset and not self._multi_roi:
            inverted_stride_dict = invert_result_dictionary(stride_lists)
            combined_stride_list = {}
            for sensor, strides in inverted_stride_dict.items():
                combined_stride_list[sensor] = self._merge_single_sensor_stride_lists(strides, index_name)

        else:
            combined_stride_list = self._merge_single_sensor_stride_lists(stride_lists, index_name)
        return combined_stride_list

    def _merge_single_sensor_stride_lists(self, stride_lists, index_name) -> StrideList:
        concat_stride_list = pd.concat(stride_lists, names=[index_name]).reset_index(index_name).reset_index(drop=True)
        concat_stride_list = concat_stride_list.sort_values("start")
        # Make the stride id unique
        if self.s_id_naming == "replace":
            concat_stride_list["s_id"] = list(range(len(concat_stride_list)))
        elif self.s_id_naming == "prefix":
            concat_stride_list["s_id"] = (
                concat_stride_list[index_name].astype(str) + "_" + concat_stride_list["s_id"].astype(str)
            )
        return concat_stride_list.set_index("s_id")

    def _validate_parameters(self) -> None:
        if self.segmentation_algorithm is None:
            raise ValueError(
                "`segmentation_algorithm` must be a valid instance of a StrideSegmentation algorithm. Currently `None`"
            )
        if self.s_id_naming not in ["replace", "prefix"]:
            raise ValueError("Invalid value for `s_id_naming`")

    def _validate_other_parameters(self) -> None:
        self._multi_dataset = is_sensor_data(self.data, check_acc=False, check_gyr=False) == "multi"
        self._multi_roi = is_regions_of_interest_list(self.regions_of_interest, region_type="any") == "multi"
        if self._multi_roi and not self._multi_dataset:
            raise ValidationError(
                "You can not use a multi-sensor regions of interest list with a single sensor dataset."
            )
        if not self._multi_roi and self._multi_dataset and isinstance(self.data, dict):
            raise ValidationError(
                "You can not use a single-sensor regions of interest list with an unsynchronised multi-sensor dataset."
            )
        if self._multi_roi and self._multi_dataset:
            sensor_names = get_multi_sensor_names(self.data)
            missing_sensors = [key for key in self.regions_of_interest if key not in sensor_names]
            if len(missing_sensors) > 0:
                raise KeyError(
                    f"The regions of interest list contains information for a sensor ({missing_sensors}) that is not "
                    "in the dataset."
                )
