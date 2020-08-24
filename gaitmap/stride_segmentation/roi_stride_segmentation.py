from typing import Dict, Hashable, Optional

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


class RoiStrideSegmentation(BaseStrideSegmentation):
    """Apply any stride segmentation algorithms to specific regions of intrest in a longer dataset."""

    segmentation_algorithm: Optional[BaseStrideSegmentation]

    instances_per_roi_: Dict[Hashable, BaseStrideSegmentation]

    data: Dataset
    sampling_rate_hz: float
    regions_of_interest: RegionsOfInterestList

    _multi_dataset: bool
    _multi_roi: bool

    def __init__(self, segmentation_algorithm: Optional[BaseStrideSegmentation] = None):
        self.segmentation_algorithm = segmentation_algorithm

    def segment(
        self: BaseType, data: Dataset, sampling_rate_hz: float, regions_of_interest: RegionsOfInterestList, **kwargs
    ) -> BaseType:
        if self.segmentation_algorithm is None:
            raise ValueError(
                "`segmentation_algorithm` must be a valid instance of a StrideSegmentation algorithm. Currently `None`"
            )

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
            for sensor, roi in self.regions_of_interest:
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
        self, sensor_data: pd.DataFrame, sampling_rate_hz: float, rois: SingleSensorRegionsOfInterestList, **kwargs
    ):
        """Call the the segmentation algorithm for each region of interest and store the instance."""
        rois = rois.reset_index()
        index_col = ROI_ID_COLS[_get_regions_of_interest_types(rois.columns)]

        instances_per_roi = {}
        combined_stride_list = {}
        for _, (index, start, end) in rois[[index_col, "start", "end"]].iterrows():
            per_roi_algo = self.segmentation_algorithm.clone()
            roi_data = sensor_data.loc[start: end]
            per_roi_algo.segment(data=roi_data, sampling_rate_hz=sampling_rate_hz, **kwargs)
            instances_per_roi[index] = per_roi_algo
            combined_stride_list[index] = per_roi_algo.stride_list_

        combined_stride_list = self._merge_stride_lists(combined_stride_list, index_col)
        return combined_stride_list, instances_per_roi

    def _merge_stride_lists(self, stride_lists: Dict[str, StrideList], index_name: str) -> StrideList:
        """Merge either single or multisensor stride lists.

        The roi id (the dict key in the input), will be a column in the final output dataframe.
        """
        # We assume all algorithms follow convention and return the correct format
        if self._multi_dataset:
            combined_stride_list = {}
            for sensor, strides in stride_lists.items():
                combined_stride_list[sensor] = pd.concat(strides, names=[index_name]).reset_index(index_name)
        else:
            combined_stride_list = pd.concat(stride_lists, names=[index_name]).reset_index(index_name)
        return combined_stride_list
