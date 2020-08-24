from typing import Dict, Hashable, Optional

import pandas as pd

from gaitmap.base import BaseStrideSegmentation, BaseType
from gaitmap.utils.dataset_helper import (
    Dataset,
    RegionsOfInterestList,
    is_single_sensor_regions_of_interest_list,
    is_multi_sensor_regions_of_interest_list,
    is_single_sensor_dataset,
    is_multi_sensor_dataset,
)


class RoiStrideSegmentation(BaseStrideSegmentation):
    """Apply any stride segmentation algorithms to specific regions of intrest in a longer dataset."""

    segmentation_algorithm: Optional[BaseStrideSegmentation]

    instances_per_roi_: Dict[Hashable, BaseStrideSegmentation]

    def __init__(self, segmentation_algorithm: Optional[BaseStrideSegmentation] = None):
        self.segmentation_algorithm = segmentation_algorithm

    def segment(
        self: BaseType, data: Dataset, sampling_rate_hz: float, regions_of_interest: RegionsOfInterestList, **kwargs
    ) -> BaseType:
        if self.segmentation_algorithm is None:
            raise ValueError(
                "`segmentation_algorithm` must be a valid instance of a StrideSegmentation algorithm. Currently `None`"
            )
        if not (
            is_single_sensor_dataset(data, check_gyr=False, check_acc=False)
            or is_multi_sensor_dataset(data, check_gyr=False, check_acc=False)
        ):
            raise ValueError("Invalid data object passed. Must either be a single or a multi sensor dataset.")
        if not (
            is_single_sensor_regions_of_interest_list(regions_of_interest)
            or is_multi_sensor_regions_of_interest_list(regions_of_interest)
        ):
            raise ValueError("Invalid object passed for `regions_of_interest`")
        if is_multi_sensor_regions_of_interest_list(regions_of_interest) and is_single_sensor_dataset(
            data, check_gyr=False, check_acc=False
        ):
            raise ValueError("You can not use a multi-sensor regions of interest list with a single sensor dataset.")
        if (
            is_single_sensor_regions_of_interest_list(regions_of_interest)
            and is_multi_sensor_dataset(data, check_gyr=False, check_acc=False)
            and isinstance(data, dict)
        ):
            raise ValueError("You can not use a single-sensor regions of interest list with an unsynchronised "
                             "multi-sensor dataset.")
        # TODO: Properly handle multi sensor ROI lists
        instances_per_roi = {}
        combined_stride_list = {}
        for index, (start, end) in regions_of_interest.iterrows():
            per_roi_algo = self.segmentation_algorithm.clone()
            per_roi_algo.segment(data=data.iloc[start, end], sampling_rate_hz=sampling_rate_hz, **kwargs)
            instances_per_roi[index] = per_roi_algo
            combined_stride_list[index] = per_roi_algo.stride_list_
        # Correctly merge multisensor stride lists
        combined_stride_list = pd.concat(combined_stride_list)
        self.stride_list_ = combined_stride_list
        self.instances_per_roi_ = instances_per_roi

        return self

    def _segment_single_sensor(self):
        """Apply the segmentation to """
        pass
