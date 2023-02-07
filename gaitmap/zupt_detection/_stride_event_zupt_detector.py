from typing import Optional

import numpy as np
import pandas as pd
from typing_extensions import Self

from gaitmap.base import BaseZuptDetector
from gaitmap.utils.datatype_helper import (
    SingleSensorData,
    SingleSensorStrideList,
    is_single_sensor_data,
    is_single_sensor_stride_list,
)
from gaitmap.utils.exceptions import ValidationError


class StrideEventZuptDetector(BaseZuptDetector):
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
    window_length_samples_
        The internally calculated window length in samples.
        This might be helpful for debugging.
    window_overlap_samples_
        The internally calculated window overlap in samples.
        This might be helpful for debugging.

    """

    half_region_size_s: float

    def __init__(self, half_region_size_s: float):
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
        is_single_sensor_data(self.data, check_acc=True, check_gyr=True, frame="any", raise_exception=True)

        try:
            is_single_sensor_stride_list(self.stride_event_list, "min_vel", raise_exception=True)
        except ValidationError as e:
            raise ValidationError(
                "For the `StrideEventZuptDetector` a proper stride_event_list of the `min_vel` type"
            ) from e

        region_size_samples = int(np.round(self.half_region_size_s * sampling_rate_hz))

        # In a min_vel stride list, all starts and all ends are min_vel events.
        all_min_vel_events = np.unique(np.concatenate([self.stride_event_list["start"], self.stride_event_list["end"]]))

        self.zupts_ = pd.DataFrame(
            {
                "start": np.clip(all_min_vel_events - region_size_samples, 0, None),
                "end": np.clip(all_min_vel_events + region_size_samples, None, self.data.shape[0]),
            }
        ).astype(int)

        return self

    @property
    def per_sample_zupts_(self) -> np.ndarray:
        """Get a bool array of length data with all Zupts as True."""
        zupts = np.zeros(self.data.shape[0], dtype=bool)
        for _, row in self.zupts_.iterrows():
            zupts[row["start"] : row["end"]] = True
        return zupts
