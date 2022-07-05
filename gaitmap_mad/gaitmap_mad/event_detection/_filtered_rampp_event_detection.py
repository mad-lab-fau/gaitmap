"""The event detection algorithm by Rampp et al. 2014."""
from typing import Dict, Optional, Tuple, Union

from joblib import Memory

from gaitmap.event_detection._event_detection_mixin import FilterParameter
from gaitmap_mad.event_detection import RamppEventDetection


class FilteredRamppEventDetection(RamppEventDetection):
    """This addition of the rampp event detection uses a 15 Hz low pass filter on the gyr-ml in ic and tc detection.

    However, the min_velocity calculation remians as standard rampp algorithm.
    It is suggested to be used on data containg high frequency noise or artifacts which affects minimum detection in
    rampp algorithm

    RamppEventDetection uses signal processing approaches to find temporal gait events by searching for characteristic
    features in the foot-mounted sensor signals as described in Rampp et al. (2014) [1]_.
    For more details refer to the `Notes` section.

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
    cutoff_frequency
        The selected cut off frequency of the low pass filter.
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
        A list of strides provided by a stride segmentatioRan method. The stride list is expected to have no gaps
        between subsequent strides. That means for subsequent strides the end sample of one stride should be the
        start sample of the next stride.

    Examples
    --------
    Get gait events from single sensor signal

    >>> event_detection = RamppEventDetection()
    >>> event_detection.detect(data=data, stride_list=stride_list, sampling_rate_hz=204.8)
    >>> event_detection.min_vel_event_list_
           start     end      ic      tc  min_vel  pre_ic
    s_id
    0      519.0   710.0   651.0   584.0    519.0   498.0
    1      710.0   935.0   839.0   802.0    710.0   651.0
    2      935.0  1183.0  1089.0  1023.0    935.0   839.0
    ...

    Notes
    -----
    Rampp et al. implemented the detection of three gait events from foot-mounted sensor data:

    terminal contact (`tc`), originally called toe-off (TO) in the paper [1]_:
        At `tc` the movement of the ankle joint changes from a plantar flexion to a dorsal extension in the sagittal
        plane.
        This change results in a zero crossing of the gyr_ml signal.
        Also refer to the :ref:`image below <fe>`.

    initial contact (`ic`), originally called heel strike (HS) in the paper [1]_:
        At `ic` the foot decelerates rapidly when the foot hits the ground.
        For the detection of `ic` only the signal between the absolute maximum and the end of the first half of the
        gyr_ml signal is considered.
        Within this segment, `ic` is found by searching for the minimum between the point of the steepest negative
        slope and the point of the steepest positive slope in the following signal.
        After that the acc_pa signal is searched for a maximum in the area before and after the described minimum in
        the gyr_ml signal.
        In the original implementation of the paper, this was actually a minimum due to flipped sensor coordinate axes.
        The default search window is set to 80 ms before and 50 ms after the minimum.
        The search window borders can be adjusted via the `ic_search_region_ms` parameter.
        Also refer to the :ref:`image below <fe>`.

    minimal velocity (`min_vel_`), originally called mid stance (MS) in the paper [1]_:
        At `min_vel` the foot has the lowest velocity.
        It is defined to be the middle of the window with the lowest energy in all axes of the gyr signal.
        The default window size is set to 100 ms with 50 % overlap.
        The window size can be adjusted via the `min_vel_search_win_size_ms` parameter.
        Also refer to the :ref:`image below <fe>`.

    The :func:`~gaitmap.event_detection.RamppEventDetection.detect` method provides a stride list `min_vel_event_list`
    with the gait events mentioned above and additionally `start` and `end` of each stride, which are aligned to the
    `min_vel` samples.
    The start sample of each stride corresponds to the min_vel sample of that stride and the end sample corresponds to
    the min_vel sample of the subsequent stride.
    Furthermore, the `min_vel_event_list` list provides the `pre_ic` which is the ic event of the previous stride in
    the stride list.

    The :class:`~gaitmap.event_detection.RamppEventDetection` includes a consistency check that is enabled by default.
    The gait events within one stride provided by the `stride_list` must occur in the expected order.
    Any stride where the gait events are detected in a different order or are not detected at all is dropped!
    For more infos on this see :func:`~gaitmap.utils.stride_list_conversion.enforce_stride_list_consistency`.
    If you wish to disable this consistency check, set `enforce_consistency` to False.
    In this case, the attribute `min_vel_event_list_` will not be set, but you can use `segmented_event_list_` to get
    all detected events for the exact stride list that was used as input.
    Note, that this list might contain NaN for some events.

    Furthermore, during the conversion from the segmented stride list to the "min_vel" stride list, breaks in
    continuous gait sequences ( with continuous subsequent strides according to the `stride_list`) are detected and the
    first (segmented) stride of each sequence is dropped.
    This is required due to the shift of stride borders between the `stride_list` and the `min_vel_event_list`.
    Thus, the first segmented stride of a continuous sequence only provides a pre_ic and a min_vel sample for
    the first stride in the `min_vel_event_list`.
    Therefore, the `min_vel_event_list` list has one stride less per gait sequence than the `segmented_stride_list`.

    Further information regarding the coordinate system can be found :ref:`here<coordinate_systems>` and regarding the
    different types of strides can be found :ref:`here<stride_list_guide>`.

    The image below gives an overview about the events and where they occur in the signal.

    .. _fe:
    .. figure:: /images/event_detection.svg

    .. [1] Rampp, A., Barth, J., Schülein, S., Gaßmann, K. G., Klucken, J., & Eskofier, B. M. (2014). Inertial
       sensor-based stride parameter calculation from gait sequences in geriatric patients. IEEE transactions on
       biomedical engineering, 62(4), 1089-1097.. https://doi.org/10.1109/TBME.2014.2368211

    """

    ic_lowpass_filter_parameter: FilterParameter

    def __init__(
        self,
        ic_search_region_ms: Tuple[float, float] = (80, 50),
        min_vel_search_win_size_ms: float = 100,
        ic_lowpass_filter_parameter: FilterParameter = FilterParameter(order=10, cutoff_hz=15),
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
