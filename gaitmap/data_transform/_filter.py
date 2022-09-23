from typing import Union, Tuple, Optional, Literal

from scipy.signal import butter, sosfiltfilt
from typing_extensions import Self

from gaitmap.data_transform._base import BaseTransformer
from gaitmap.utils.datatype_helper import SingleSensorData


class BaseFilter(BaseTransformer):
    _action_methods = (*BaseTransformer._action_methods, "filter")

    @property
    def filtered_signal_(self) -> SingleSensorData:
        return self.transformed_data_

    def filter(self, data: SingleSensorData, *, sampling_rate_hz: Optional[float] = None, **kwargs):
        """Filter the data.

        This is identical to the transform method, but is here for conveience, as it is more natural to type
        filter.filter() instead of filter.transform().
        """
        return self.transform(data, sampling_rate_hz=sampling_rate_hz, **kwargs)


class ButterworthFilter(BaseFilter):
    """Apply a forward-backward (filtfilt) butterworth filter using the transformer interface."""

    order: int
    cutoff_freq_hz: Union[float, Tuple[float, float]]
    filter_type: Literal["lowpass", "highpass", "bandpass", "bandstop"]

    def __int__(
        self,
        order: int,
        cutoff_freq_hz: Union[float, Tuple[float, float]],
        filter_type: Literal["lowpass", "highpass", "bandpass", "bandstop"] = "lowpass",
    ):
        self.order = order
        self.cutoff_freq_hz = cutoff_freq_hz
        self.filter_type = filter_type

    def transform(self, data: SingleSensorData, *, sampling_rate_hz: Optional[float] = None, **kwargs) -> Self:
        if sampling_rate_hz is None:
            raise ValueError(f"{type(self).__name__}.transform requires a `sampling_rate_hz` to be passed.")

        sos = butter(self.order, self.cutoff_freq_hz, btype=self.filter_type, output="sos", fs=sampling_rate_hz)
        self.transformed_data_ = sosfiltfilt(sos=sos, x=data)
        return self
