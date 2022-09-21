"""The event detection algorithm by Rampp et al. 2014."""
from typing import Dict, Optional, Tuple, Union

from joblib import Memory

from gaitmap.event_detection._event_detection_mixin import FilterParameter
from gaitmap_mad.event_detection._rampp_event_detection import RamppEventDetection


class FilteredRamppEventDetection(RamppEventDetection):
    """This addition uses a low-pass filter on the ml signal for ic calculation.

    Rampp event detection with an additional low-pass Butterworth filter.
    This method is suggested to be used on data containing high frequency noise or artifacts which occur around the ic
    part of the strides.
    The tc and min velocity are calculated from the original signal.

    Parameters
    ----------
    ic_search_region_ms
        The region to look for the initial contact in the acc_pa signal in ms given an ic candidate. According to [1]_,
        for the ic the algorithm first looks for a local minimum in the gyr_ml signal after the swing phase. The actual
        ic is then determined in the acc_pa signal in the ic_search_region_ms around that gyr_ml minimum.
        ic_search_region_ms[0] describes the start and ic_search_region_ms[1] the end of the region to check around the
        gyr_ml minimum. The values of `ic_search_region_ms` must be greater or equal than the sample time
        (1/`sampling_rate_hz`).
    min_vel_search_win_size_ms
        The size of the sliding window for finding the minimum gyroscope energy in ms.
    ic_lowpass_filter_parameter
        A tuple including the information required for a low pass filter design in the format of
        (order, cutoff frequency).
    memory
        An optional `joblib.Memory` object that can be provided to cache the detection of all events.
    enforce_consistency
        An optional bool that can be set to False if you wish to disable postprocessing
        (see Notes section for more information).

    Attributes
    ----------
    min_vel_event_list_ : A stride list or dictionary with such values
        The result of the `detect` method holding all temporal gait events and start / end of all strides.
        The stride borders for the stride_events are aligned with the min_vel samples.
        Hence, the start sample of each stride corresponds to the min_vel sample of that stride and the end sample
        corresponds to the min_vel sample of the subsequent stride.
        Strides for which no valid events could be found are removed.
        Additional strides might have been removed due to the conversion from segmented to min_vel strides.
        The 's_id' index is selected according to which segmented stride the pre-ic belongs to.

    segmented_event_list_ : A stride list or dictionary with such values
        The result of the `detect` method holding all temporal gait events and start / end of all strides.
        This version of the results has the same stride borders than the input `stride_list` and has additional columns
        for all the detected events.
        Strides for which no valid events could be found are removed.

    Other Parameters
    ----------------
    data
        The data passed to the `detect` method.
    sampling_rate_hz
        The sampling rate of the data
    stride_list
        A list of strides provided by a stride segmentation method. The stride list is expected to have no gaps
        between subsequent strides. That means for subsequent strides the end sample of one stride should be the
        start sample of the next stride.

    Notes
    -----
    Due to attachment methods used for foot-worn IMUs, the sensor might experience bounce and vibrate at the time of
    the heel strike (IC) which leads to high frequency artifacts in the gyr_ml signal.
    This can lead to an inaccurate IC detection, as it relies on the identification of extrema in the signal.
    To resolve this issue, this event detection method applies a low-pass filter to remove high frequency artifacts.
    Other than that, the implementation is identical to the normal Rampp event detection (
    :class:`~gaitmap.event_detection.RamppEventDetection`).

    .. [1] Rampp, A., Barth, J., Schülein, S., Gaßmann, K. G., Klucken, J., & Eskofier, B. M. (2014). Inertial
       sensor-based stride parameter calculation from gait sequences in geriatric patients. IEEE transactions on
       biomedical engineering, 62(4), 1089-1097.. https://doi.org/10.1109/TBME.2014.2368211

    """

    ic_lowpass_filter_parameter: FilterParameter

    def __init__(
        self,
        ic_search_region_ms: Tuple[float, float] = (80, 50),
        min_vel_search_win_size_ms: float = 100,
        ic_lowpass_filter_parameter: FilterParameter = FilterParameter(order=2, cutoff_hz=15),
        memory: Optional[Memory] = None,
        enforce_consistency: bool = True,
    ):
        self.ic_lowpass_filter_parameter = ic_lowpass_filter_parameter
        super().__init__(
            memory=memory,
            enforce_consistency=enforce_consistency,
            ic_search_region_ms=ic_search_region_ms,
            min_vel_search_win_size_ms=min_vel_search_win_size_ms,
        )

    def _get_detect_kwargs(self) -> Dict[str, Union[Tuple[int, int], int]]:  # noqa: no-self-use
        parent_kwargs = super()._get_detect_kwargs()
        return {**parent_kwargs, "gyr_ic_lowpass_filter_parameters": self.ic_lowpass_filter_parameter}
