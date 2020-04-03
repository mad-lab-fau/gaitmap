"""The event detection algorithm by Rampp et al. 2014."""
from typing import Optional, Sequence, List

import numpy as np
import pandas as pd

from gaitmap.base import BaseEventDetection, BaseType


class RamppEventDetection(BaseEventDetection):
    """Find gait events in the IMU raw signal based on signal characteristics like peaks.

    BarthDtw uses a manually created template of an IMU stride to find multiple occurrences of similar signals in a
    continuous data stream.
    The method is not limited to a single sensor-axis or sensor, but can use any number of dimensions of the provided
    input signal simultaneously.
    For more details refer to the `Notes` section.

    Attributes
    ----------
    TODO add attributes

    Parameters
    ----------
    TODO add parameters

    Other Parameters
    ----------------
    TODO add other paramters

    Notes
    -----
    TODO: Add additional details about the use of DTW for stride segmentation

    [1] Rampp, A., Barth, J., Schülein, S., Gaßmann, K. G., Klucken, J., & Eskofier, B. M. (2014). Inertial
    sensor-based stride parameter calculation from gait sequences in geriatric patients. IEEE transactions on biomedical
    engineering, 62(4), 1089-1097.. https://doi.org/10.1109/TBME.2014.2368211

    """

    data: pd.DataFrame
    sampling_rate_hz: float
    stride_list: pd.DataFrame

    def __init__(self):
        pass

    def detect(self: BaseType, data: pd.DataFrame, sampling_rate_hz: float, stride_list: pd.DataFrame) -> BaseType:
        """Find gait events in data within strides provided by stride_list.

    Parameters
    ----------
    TODO parameters
    data
    sampling_rate_hz
        The sampling rate of the data signal. This will be used to convert all parameters provided in seconds into
        a number of samples and it will be used to resample the template if `resample_template` is `True`.
    stride_list

    Returns
    -------
        self
            The class instance with all result attributes populated

    """

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        return self
