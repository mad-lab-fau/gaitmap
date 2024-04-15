from typing import List, Literal, Optional, Tuple

import numpy as np
from typing_extensions import Self

from gaitmap.base import BaseZuptDetector
from gaitmap.utils.datatype_helper import SingleSensorData
from gaitmap.zupt_detection._base import PerSampleZuptDetectorMixin


class ComboZuptDetector(BaseZuptDetector, PerSampleZuptDetectorMixin):
    """A ZUPT detector that combines multiple ZUPT detectors.

    Parameters
    ----------
    detectors
        A list of tuples of the form `(name, detector_instance)`.
    operation
        The operation to combine the detectors.
        Must be one of `and`, `or`.

    Other Parameters
    ----------------
    data
        The data passed to the detect method
    sampling_rate_hz
        The sampling rate of this data

    Attributes
    ----------
    zupts_
        A dataframe with the columns `start` and `end` specifying the start and end of all static regions in samples
    per_sample_zupts_
        A bool array with length `len(data)`.
        If the value is `True` for a sample, it is part of a static region.
    min_vel_value_
        Always None. Only implemented for API compatibility.
    min_vel_index_
        Always None. Only implemented for API compatibility.

    """

    _composite_params = ("detectors",)

    detectors: Optional[List[Tuple[str, BaseZuptDetector]]]
    operation: Literal["and", "or"]

    def __init__(
        self, detectors: Optional[List[Tuple[str, BaseZuptDetector]]] = None, operation: Literal["and", "or"] = "or"
    ) -> None:
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
            The class instance with all result attributes populated

        """
        if not self.detectors:
            raise ValueError("No detectors have been set.")

        self.sampling_rate_hz = sampling_rate_hz
        self.data = data

        # Note, we don't validate the data. If any of the ZUPT detectors need it, they will do it themselves.
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

        # Set for API compatibility
        self.min_vel_value_ = None
        self.min_vel_index_ = None

        return self
