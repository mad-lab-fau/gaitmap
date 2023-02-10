from typing import Optional

import numpy as np
import pandas as pd
from typing_extensions import Self

from gaitmap.base import BaseZuptDetector
from gaitmap.utils.array_handling import merge_intervals
from gaitmap.utils.datatype_helper import (
    SingleSensorData,
    SingleSensorStrideList,
    is_single_sensor_data,
    is_single_sensor_stride_list,
)
from gaitmap.utils.exceptions import ValidationError
from gaitmap.zupt_detection._base import RegionZuptDetectorMixin


class StrideEventZuptDetector(BaseZuptDetector, RegionZuptDetectorMixin):
    """A ZUPT detector that simply reuses the min_vel events as ZUPT events.

    This can be very helpful, when wanting to enforce one ZUPT event per stride.
    The data is actually ignored completely and only the event list passed to the detect method is used.

    Parameters
    ----------
    half_region_size_s
        Half the size of the region around the min_vel event that is considered a ZUPT event in seconds.
        The region from `min_vel_event - half_region_size_s` to `min_vel_event + half_region_size_s` is considered a
        ZUPT event.

    Other Parameters
    ----------------
    data
        The data passed to the detect method
    stride_event_list
        The stride event list passed to the detect method
    sampling_rate_hz
        The sampling rate of this data

    Attributes
    ----------
    zupts_
        A dataframe with the columns `start` and `end` specifying the start and end of all static regions in samples
    per_sample_zupts_
        A bool array with length `len(data)`.
        If the value is `True` for a sample, it is part of a static region.
    half_region_size_samples_
        The actual half region size in samples calculated using the data sampling rate.
    min_vel_value_
        Always None. Only implemented for API compatibility.
    min_vel_index_
        Always None. Only implemented for API compatibility.

    """

    half_region_size_s: float
    half_region_size_samples_: int

    def __init__(self, half_region_size_s: float = 0.05):
        self.half_region_size_s = half_region_size_s

    def detect(
        self,
        data: SingleSensorData,
        *,
        stride_event_list: Optional[SingleSensorStrideList] = None,
        sampling_rate_hz: float,
    ) -> Self:
        """Detect the ZUPT events using the stride event list.

        Parameters
        ----------
        data
            The data set holding the imu raw data.
            The data is ignored completly during the calculation.
        stride_event_list
            The stride event list to use for the detection.
            This must be a min_vel stride event list (i.e. all strides should start and end with a min_vel event).
        sampling_rate_hz
            The sampling rate of the data

        Returns
        -------
        self
            The class instance with all result attributes populated

        """
        self.data = data
        self.stride_event_list = stride_event_list
        self.sampling_rate_hz = sampling_rate_hz

        if self.half_region_size_s < 0:
            raise ValueError("The half region size must be >= 0")

        # We don't need the data. We still check it, as we need its length for the per_sample_zupts_ attribute.
        # This means, we need to make at least sure that the data is somewhat valid.
        is_single_sensor_data(self.data, check_acc=False, check_gyr=False, frame="any", raise_exception=True)

        try:
            is_single_sensor_stride_list(
                self.stride_event_list, "min_vel", check_additional_cols=("min_vel",), raise_exception=True
            )
        except ValidationError as e:
            raise ValidationError(
                "For the `StrideEventZuptDetector` a proper stride_event_list of the `min_vel` type is required."
            ) from e

        self.half_region_size_samples_ = int(np.round(self.half_region_size_s * sampling_rate_hz))

        # In a min_vel stride list, all starts and all ends are min_vel events.
        all_min_vel_events = np.unique(np.concatenate([self.stride_event_list["start"], self.stride_event_list["end"]]))

        start_ends = np.empty((len(all_min_vel_events), 2), dtype=int)
        start_ends[:, 0] = np.clip(all_min_vel_events - self.half_region_size_samples_, 0, None)
        start_ends[:, 1] = np.clip(all_min_vel_events + self.half_region_size_samples_ + 1, None, self.data.shape[0])
        self.zupts_ = pd.DataFrame(merge_intervals(start_ends), columns=["start", "end"])
        # This is required, because otherwise, edge cases at the start or end of the data could lead to zero-length
        # ZUPTs.
        self.zupts_ = self.zupts_.loc[self.zupts_["start"] < self.zupts_["end"]]

        # Set for API compatibility
        self.min_vel_value_ = None
        self.min_vel_index_ = None

        return self
