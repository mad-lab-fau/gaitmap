"""The gait sequence detection algorithm by Rampp et al. 2020."""
from typing import Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd
from numpy.linalg import norm

from gaitmap.base import BaseGaitDetection, BaseType
from gaitmap.utils.array_handling import sliding_window_view
from gaitmap.utils.consts import BF_ACC, BF_GYR
from gaitmap.utils.dataset_helper import (
    is_multi_sensor_dataset,
    is_single_sensor_dataset,
    is_single_sensor_stride_list,
    is_multi_sensor_stride_list,
    StrideList,
    Dataset,
    get_multi_sensor_dataset_names,
)


class UllrichGaitSequenceDetection(BaseGaitDetection):
    """Find gait events in the IMU raw signal based on signal characteristics.

    UllrichGaitSequenceDetection uses signal processing approaches to find gait sequences by searching for
    characteristic features in the power spectral density of the imu raw signals as described in Ullrich et al. (
    2020)  [  1]_.
    For more details refer to the `Notes` section.

    Parameters
    ----------
    TODO add parameters

    Attributes
    ----------
    TODO finalize
    TODO do we need a datatype 'gait_sequence_list'? Can this be a subtype of a stride_list?
    gait_sequences_ : A gait_sequence_list or dictionary with such values
        The result of the `detect` method holding all gait sequences with their start and end samples. Formatted
        as pandas DataFrame.
    start_ : 1D array or dictionary with such values
        The array of start samples of all gait_sequences. Start samples of each gait sequence correspond to the start
        sample of the window in which it was found.
    end_ : 1D array or dictionary with such values
        The array of end samples of all strides. End samples of each gait sequence correspond to the start sample of the
        window in which it was found.

    Other Parameters
    ----------------
    data
        The data passed to the `detect` method.
    sampling_rate_hz
        The sampling rate of the data

    Notes
    -----
    TODO: Add additional details about the algorithm for event detection

    .. [1] M. Ullrich, A. Küderle, J. Hannink, S. Del Din, H. Gassner, F. Marxreiter, J. Klucken, B.M.
        Eskofier, F. Kluge, Detection of Gait From Continuous Inertial Sensor Data Using Harmonic
        Frequencies, IEEE Journal of Biomedical and Health Informatics. (2020) 1–1.
        https://doi.org/10.1109/JBHI.2020.2975361.

    """

    sensor_channel_config: str
    peak_prominence: float
    window_size_s: float

    start_: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
    end_: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
    gait_sequences_: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]

    data: Dataset
    sampling_rate_hz: float

    def __init__(self, sensor_channel_config: str = "gyr_ml", peak_prominence: float = 17, window_size_s: float = 10):
        self.sensor_channel_config = sensor_channel_config
        self.peak_prominence = peak_prominence
        self.window_size_s = window_size_s

    def detect(self: BaseType, data: Dataset, sampling_rate_hz: float) -> BaseType:
        """Find gait sequences in data.

        Parameters
        ----------
        data
            The data set holding the imu raw data
        sampling_rate_hz
            The sampling rate of the data

        Returns
        -------
        self
            The class instance with all result attributes populated

        Examples
        --------
        TODO add example

        """

        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        # TODO check for reasonable input values for sensor_channel_config, peak_prominence and window_size_s
        # sensor_channel_config can be list or str. list must be == BF_ACC/GYR, str must be one of it entries

        window_size = self.window_size_s * self.sampling_rate_hz

        if is_single_sensor_dataset(data):
            (self.gait_sequences_, self.start_, self.end_,) = self._detect_single_dataset(data, window_size)
        elif is_multi_sensor_dataset(data):
            self.gait_sequences_ = dict()
            self.start_ = dict()
            self.end_ = dict()
            # TODO merge the gait sequences from both feet? with logical or?
            for sensor in get_multi_sensor_dataset_names(data):
                (self.gait_sequences_[sensor], self.start_[sensor], self.end_[sensor],) = self._detect_single_dataset(
                    data[sensor], window_size
                )
        else:
            raise ValueError("Provided data set is not supported by gaitmap")

        return self

    def _detect_single_dataset(
        self, data: pd.DataFrame, window_size: float,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Detect gait sequences for a single sensor data set."""

        # define 3d signal to analyze for active signal and further parameters
        if "acc" in self.sensor_channel_config:
            s_3d = data[BF_ACC]
            active_signal_th = 0.2
            fft_factor = 100
        else:
            s_3d = data[BF_GYR]
            active_signal_th = 50
            fft_factor = 1

        # define the 1d signal to analyze for frequency spectrum
        if type(self.sensor_channel_config) == list:
            s_1d = norm(np.array(data[self.sensor_channel_config]), axis=1)
        else:
            s_1d = data[self.sensor_channel_config]

        # TODO put awesome code here
        gait_sequences_ = None
        start_ = None
        end_ = None

        return gait_sequences_, start_, end_
