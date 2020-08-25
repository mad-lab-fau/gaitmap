from typing import Dict, Hashable, Optional, TypeVar, Generic
from typing_extensions import Literal

import pandas as pd

from gaitmap.base import BaseStrideSegmentation, BaseType
from gaitmap.utils.consts import ROI_ID_COLS
from gaitmap.utils.dataset_helper import (
    Dataset,
    RegionsOfInterestList,
    is_single_sensor_regions_of_interest_list,
    is_multi_sensor_regions_of_interest_list,
    is_single_sensor_dataset,
    is_multi_sensor_dataset,
    SingleSensorRegionsOfInterestList,
    _get_regions_of_interest_types,
    StrideList,
)

StrideSegmentationAlgorithm = TypeVar("StrideSegmentationAlgorithm", bound=BaseStrideSegmentation)


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
    :ref:`coordinate system guide <coordinate_systems>`.

    Parameters
    ----------
    segmentation_algorithm
        An instance of a valid segmentation algorithm with all the wanted parameters set.

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

    instances_per_roi_: Dict[Hashable, StrideSegmentationAlgorithm]
    stride_list_: StrideList

    data: Dataset
    sampling_rate_hz: float
    regions_of_interest: RegionsOfInterestList

    _multi_dataset: bool
    _multi_roi: bool

    def __init__(
        self,
        segmentation_algorithm: Optional[StrideSegmentationAlgorithm] = None,
        s_id_naming: Literal["replace", "prefix"] = "replace",
    ):
        self.segmentation_algorithm = segmentation_algorithm
        self.s_id_naming = s_id_naming

    def segment(
        self: BaseType, data: Dataset, sampling_rate_hz: float, regions_of_interest: RegionsOfInterestList, **kwargs
    ) -> BaseType:
        if self.segmentation_algorithm is None:
            raise ValueError(
                "`segmentation_algorithm` must be a valid instance of a StrideSegmentation algorithm. Currently `None`"
            )
        if self.s_id_naming not in ["replace", "prefix"]:
            raise ValueError("Invalid value for `s_id_naming`")
        if is_single_sensor_dataset(data, check_gyr=False, check_acc=False):
            self._multi_dataset = False
        elif is_multi_sensor_dataset(data, check_gyr=False, check_acc=False):
            self._multi_dataset = True
        else:
            raise ValueError("Invalid data object passed. Must either be a single or a multi sensor dataset.")
        if is_single_sensor_regions_of_interest_list(regions_of_interest):
            self._multi_roi = False
        elif is_multi_sensor_regions_of_interest_list(regions_of_interest):
            self._multi_roi = True
        else:
            raise ValueError("Invalid object passed for `regions_of_interest`")
        if self._multi_roi and not self._multi_dataset:
            raise ValueError("You can not use a multi-sensor regions of interest list with a single sensor dataset.")
        if not self._multi_roi and self._multi_dataset and isinstance(data, dict):
            raise ValueError(
                "You can not use a single-sensor regions of interest list with an unsynchronised "
                "multi-sensor dataset."
            )

        self.regions_of_interest = regions_of_interest
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz
        if self._multi_roi:
            # Apply the segmentation to a single dataset in case a multi - sensor roi list is provided
            stride_list = {}
            instances = {}
            for sensor, roi in self.regions_of_interest.items():
                try:
                    sensor_data = self.data[sensor]
                except KeyError:
                    raise KeyError(
                        "The regions of interest list contains information for a sensor ({}) that is not in the "
                        "dataset.".format(sensor)
                    )
                combined_stride_list, instances_per_roi = self._segment_single_sensor(
                    sensor_data, self.sampling_rate_hz, roi, **kwargs
                )
                stride_list[sensor] = combined_stride_list
                instances[sensor] = instances_per_roi
            self.stride_list_ = stride_list
            self.instances_per_roi_ = instances
        else:
            combined_stride_list, instances_per_roi = self._segment_single_sensor(
                self.data, self.sampling_rate_hz, self.regions_of_interest, **kwargs
            )

            self.stride_list_ = combined_stride_list
            self.instances_per_roi_ = instances_per_roi

        return self

    def _segment_single_sensor(
        self,
        sensor_data: pd.DataFrame,
        sampling_rate_hz: float,
        rois: SingleSensorRegionsOfInterestList,
        sensor_name: Optional[str] = None,
        **kwargs,
    ):
        """Call the the segmentation algorithm for each region of interest and store the instance."""
        rois = rois.reset_index()
        index_col = ROI_ID_COLS[_get_regions_of_interest_types(rois.columns)]

        instances_per_roi = {}
        combined_stride_list = {}
        for _, (index, start, end) in rois[[index_col, "start", "end"]].iterrows():
            per_roi_algo = self.segmentation_algorithm.clone()
            roi_data = sensor_data.iloc[start:end]
            if sensor_name:
                # In case the original dataset was a dict, sensor_name is passed by the parent.
                # Here we ensure that the data passed into segment method is still a dict to avoid potential issues with
                # specific segmentation methods
                roi_data = {sensor_name: roi_data}
            per_roi_algo.segment(data=roi_data, sampling_rate_hz=sampling_rate_hz, **kwargs)
            instances_per_roi[index] = per_roi_algo
            stride_list = per_roi_algo.stride_list_
            if isinstance(stride_list, dict):
                for strides in stride_list.values():
                    # This is an inplace modification!
                    strides[["start", "end"]] += start
            else:
                stride_list[["start", "end"]] += start
            if sensor_name:
                combined_stride_list[index] = per_roi_algo.stride_list_[sensor_name]
            combined_stride_list[index] = per_roi_algo.stride_list_

        combined_stride_list = self._merge_stride_lists(combined_stride_list, index_col)
        return combined_stride_list, instances_per_roi

    def _merge_stride_lists(self, stride_lists: Dict[str, StrideList], index_name: str) -> StrideList:
        """Merge either single or multisensor stride lists.

        The roi id (the dict key in the input), will be a column in the final output dataframe.
        """
        # We assume all algorithms follow convention and return the correct format
        if self._multi_dataset and not self._multi_roi:
            inverted_stride_dict = {}
            for roi, stride_list in stride_lists.items():
                for sensor, strides in stride_list.items():
                    inverted_stride_dict.setdefault(sensor, {})[roi] = strides
            combined_stride_list = {}
            for sensor, strides in inverted_stride_dict.items():
                combined_stride_list[sensor] = self._merge_single_sensor_stride_lists(strides, index_name)

        else:
            combined_stride_list = self._merge_single_sensor_stride_lists(stride_lists, index_name)
        return combined_stride_list

    def _merge_single_sensor_stride_lists(self, stride_lists, index_name) -> StrideList:
        concat_stride_list = pd.concat(stride_lists, names=[index_name]).reset_index(index_name).reset_index(drop=True)
        # Make the stride id unique
        if self.s_id_naming == "replace":
            concat_stride_list["s_id"] = list(range(len(concat_stride_list)))
        elif self.s_id_naming == "prefix":
            concat_stride_list["s_id"] = (
                concat_stride_list[index_name].astype(str) + "_" + concat_stride_list["s_id"].astype(str)
            )
        return concat_stride_list.set_index("s_id")
