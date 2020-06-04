"""The gait sequence detection algorithm by Rampp et al. 2020."""
from typing import Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd
from numpy.linalg import norm

from scipy.signal import butter, lfilter, find_peaks

from gaitmap.base import BaseGaitDetection, BaseType
from gaitmap.utils.array_handling import sliding_window_view
from gaitmap.utils.consts import BF_ACC, BF_GYR
from gaitmap.utils.dataset_helper import (
    is_multi_sensor_dataset,
    is_single_sensor_dataset,
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
        # todo add flag to merge resulting gait sequences, set to False
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

        window_size = int(self.window_size_s * self.sampling_rate_hz)

        if is_single_sensor_dataset(data):
            (self.gait_sequences_, self.start_, self.end_,) = self._detect_single_dataset(data, window_size)
        elif is_multi_sensor_dataset(data):
            self.gait_sequences_ = dict()
            self.start_ = dict()
            self.end_ = dict()
            # TODO merge the gait sequences from both feet? with logical or?
            # TODO check if dataframe (=synced) or dict of dataframes, if synced set merge flat to true
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
        gait_sequences_dict = {"start": [], "end": []}

        # define 3d signal to analyze for active signal and further parameters
        if "acc" in self.sensor_channel_config:
            s_3d = np.array(data[BF_ACC])
            active_signal_th = 0.2
            fft_factor = 100
        else:
            s_3d = np.array(data[BF_GYR])
            active_signal_th = 50
            fft_factor = 1

        # subtract mean to compensate for gravity in case of acc / bad calibration in case of gyr
        s_3d_zero_mean = s_3d - np.mean(s_3d, axis=0)
        s_3d_norm = norm(s_3d_zero_mean, axis=1)

        # define the 1d signal to analyze for frequency spectrum
        if isinstance(self.sensor_channel_config, list):
            s_1d = s_3d_norm
        else:
            s_1d = np.array(data[self.sensor_channel_config])
            s_1d = s_1d - np.mean(s_1d, axis=0)

        # sliding windows
        overlap = int(window_size / 2)
        s_3d_norm = sliding_window_view(s_3d_norm, window_size, overlap)
        s_1d = sliding_window_view(s_1d, window_size, overlap)

        # active signal detection
        # create boolean mask
        active_signal_mask = np.mean(s_3d_norm, axis=1) > active_signal_th
        s_3d_norm = s_3d_norm[active_signal_mask, :]
        s_1d = s_1d[active_signal_mask, :]

        row_idx = 0
        for row_norm, row_s_1d in zip(s_3d_norm, s_1d):

            if self._ullrich_gsd_algorithm(row_norm, row_s_1d, fft_factor, active_signal_th):
                gait_sequences_dict["start"].append(row_idx * overlap)
                gait_sequences_dict["end"].append(row_idx * overlap + window_size)

            row_idx = row_idx + 1

        gait_sequences_ = pd.DataFrame(gait_sequences_dict)
        start_ = np.array(gait_sequences_dict["start"])
        end_ = np.array(gait_sequences_dict["end"])

        # todo concat overlapping gs

        return gait_sequences_, start_, end_

    def _ullrich_gsd_algorithm(self, s_3d_norm, s_1d, fft_factor, active_signal_th):
        """Apply the actual algorithm with the single processing steps."""
        gait_sequence_flag = True

        # lowpass filtering
        # TODO filter before windowing
        lp_freq_hz = 6  # 6 Hz as harmonics are supposed to occur in lower frequencies
        s_1d = _butter_lowpass_filter(s_1d, lp_freq_hz, self.sampling_rate_hz)

        # apply FFT
        frq_axis, f_s_1d = _my_fft(s_1d, self.sampling_rate_hz)
        f_s_1d = f_s_1d * fft_factor

        # find dominant frequency
        dominant_frq = _autocorr(s_1d, frq_axis, self.sampling_rate_hz)

        if dominant_frq < 0.5:
            gait_sequence_flag = False
        else:
            # apply peak detection: peaks should be higher than mean of f_s_1d
            min_height = np.mean(f_s_1d[frq_axis <= lp_freq_hz])
            peaks, _ = find_peaks(
                f_s_1d[frq_axis <= lp_freq_hz], height=min_height, prominence=self.peak_prominence
            )
            # find index of dominant frq on frq_axis
            dominant_frq_idx = np.where(frq_axis == dominant_frq)[0]

            # define harmonics candidates as 4 multiples of dominant_frq_idx
            harmonics_candidates = np.array([dominant_frq_idx[0] * factor for factor in range(2, 6)])

            # get delta in Hz on frq_axis
            frq_axis_delta = frq_axis[1]

            # define size of window around candidates to look for peak. Allow 0.3 Hz of tolerance
            # TODO expose tolerance as hidden class attribute
            harmonic_window_half = int(np.floor(0.3 / frq_axis_delta))

            # list to collect decision of harmonic or not
            candidate_evaluation = np.array([False for candidate in harmonics_candidates])

            for i_candidate, candidate in enumerate(harmonics_candidates):
                # if the candidate is at the position of a detected peak assign to be harmonic
                if np.isin(candidate, peaks):
                    candidate_evaluation[i_candidate] = True
                # if not, look for a peak in a window around the candidate
                else:
                    # define window
                    window = np.arange(candidate - harmonic_window_half, candidate + harmonic_window_half + 1)
                    # look for matches between window indices and detected peaks
                    peak_matches = np.where(np.isin(window, peaks))[0]
                    # if there is exactly one match, this is assigned to be harmonic
                    if peak_matches.size == 1:
                        # only assign if the value is above the minimum height for a peak
                        if f_s_1d[peak_matches[0] + window[0]] > min_height:
                            harmonics_candidates[i_candidate] = peak_matches[0] + window[0]
                            candidate_evaluation[i_candidate] = True
                        else:
                            candidate_evaluation[i_candidate] = False
                    # if there is no match at all, there is no harmonic detected
                    elif peak_matches.size == 0:
                        candidate_evaluation[i_candidate] = False
                    # if there is more than one match, decide for the peak with the highest value
                    else:
                        window_max = np.max(f_s_1d[window])
                        # only assign if the value is above the minimum height for a peak
                        if window_max >= min_height:
                            harmonics_candidates[i_candidate] = np.argmax(f_s_1d[window]) + window[0]
                            candidate_evaluation[i_candidate] = True
                        else:
                            candidate_evaluation[i_candidate] = False

                # if the majority of decisions is pro harmonic, set the boutFlag to true
            if np.where(~candidate_evaluation)[0].size <= len(candidate_evaluation) / 2:
                gait_sequence_flag = True
                harmonics_candidates = harmonics_candidates[candidate_evaluation]
            else:
                gait_sequence_flag = False
                harmonics_candidates = harmonics_candidates[candidate_evaluation]

        return gait_sequence_flag


# TODO consistent naming of variables in helper functions
def _butter_lowpass_filter(data, cutoff, sampling_rate_hz, order=4):
    """Create and apply butterworth lowpass filter."""
    nyq = 0.5 * sampling_rate_hz
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = lfilter(b, a, data)
    return y


def _my_fft(sig, sampling_rate_hz):
    """Perform a fft and also return the frequency axis in Hz."""
    n = len(sig)  # length of the signal
    k = np.arange(n)
    t_end = (n * 1.0) / sampling_rate_hz
    frq = (k * 1.0) / t_end  # two sides frequency range
    frq = frq[range(n // 2)]  # one side frequency range

    y = np.fft.fft(sig, n) / n  # fft computing and normalization
    y = y[range(n // 2)]

    return frq, abs(y)


def _autocorr(x, frq_axis, sampling_rate_hz):
    """Compute autocorrelation and find dominant frequency peak."""
    result = np.correlate(x, x, mode="full")
    # TODO expose range limits as hidden class attributes
    # TODO check: 5 * upper range limit should be << nyquist frq
    lower_bound = int(np.floor(sampling_rate_hz / 3))  # (=102.4 Hz / 3 Hz = upper bound of locomotor band)
    upper_bound = int(np.ceil(sampling_rate_hz / 0.5))  # (=102.4 Hz / 0.5 Hz = lower bound of locomotor band)

    result_half = result[result.size // 2 :]

    dominant_peak = np.argmax(result_half[lower_bound : upper_bound + 1]) + lower_bound
    dominant_frequency = sampling_rate_hz / dominant_peak

    dominant_frequency = frq_axis[(np.abs(frq_axis - dominant_frequency)).argmin()]

    return dominant_frequency
