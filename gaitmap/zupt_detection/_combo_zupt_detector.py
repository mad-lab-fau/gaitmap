from typing import List, Literal, Tuple

import numpy as np
from typing_extensions import Self

from gaitmap.base import BaseZuptDetector
from gaitmap.utils.datatype_helper import SingleSensorData, is_single_sensor_data
from gaitmap.zupt_detection._moving_window_zupt_detector import _PerSampleDetectorMixin


class ComboZuptDetector(BaseZuptDetector, _PerSampleDetectorMixin):
    """A ZUPT detector that combines multiple ZUPT detectors.

    Parameters
    ----------
    detectors
        A list of tuples of the form `(name, detector_instance)`.
    operation
        The operation to combine the detectors.
        Must be one of `and`, `or`.

    """

    _composite_params = ("detectors",)

    def __init__(self, detectors: List[Tuple[str, BaseZuptDetector]], operation: Literal["and", "or"] = "or"):
        self.detectors = detectors
        self.operation = operation

    def detect(
        self,
        data: SingleSensorData,
        *,
        sampling_rate_hz: float,
        **kwargs,
    ) -> Self:
        """Detect static regions in the data.

        Parameters
        ----------
        data
            The data to detect static regions in
        sampling_rate_hz
            The sampling rate of the data

        Returns
        -------
        self
            The fitted instance
        """
        self.sampling_rate_hz = sampling_rate_hz
        self.data = data

        is_single_sensor_data(self.data, frame="any", check_acc=True, check_gyr=True, raise_exception=True)

        single_zupts = []
        for _name, detector in self.detectors:
            detector = detector.clone().detect(data, sampling_rate_hz=sampling_rate_hz, **kwargs)
            single_zupts.append(detector.per_sample_zupts_)

        if self.operation == "or":
            self.per_sample_zupts_ = np.logical_or.reduce(single_zupts)
        elif self.operation == "and":
            self.per_sample_zupts_ = np.logical_and.reduce(single_zupts)
        else:
            raise ValueError(f"Unknown operation `{self.operation}` to combine detectors. Must be one of `and`, `or`.")

        return self
