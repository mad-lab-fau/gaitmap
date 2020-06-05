"""The gait sequence detection algorithm by Rampp et al. 2020."""
from typing import Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.fft import rfft

from scipy.signal import butter, lfilter, peak_prominences

from numba import njit

from gaitmap.base import BaseGaitDetection, BaseType
from gaitmap.utils.array_handling import sliding_window_view, find_extrema_in_radius
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
        s_3d_norm, s_1d, active_signal_th, fft_factor = self._signal_extraction(data)

        # sig_length is required later for the concatenation of gait sequences
        sig_length = len(s_1d)

        # lowpass filter the signal
        # TODO this is now happening before the windowing and thus before the active signal detection. Does this
        #  change the results?
        lp_freq_hz = 6  # 6 Hz as harmonics are supposed to occur in lower frequencies
        s_1d = _butter_lowpass_filter(s_1d, lp_freq_hz, self.sampling_rate_hz)

        # sliding windows
        overlap = int(window_size / 2)
        s_3d_norm = sliding_window_view(s_3d_norm, window_size, overlap)
        s_1d = sliding_window_view(s_1d, window_size, overlap)

        # active signal detection
        s_1d, active_signal_mask = self._active_signal_detection(s_3d_norm, s_1d, active_signal_th)

        # dominant frequency via autocorrelation
        dominant_frequency = self._get_dominant_frequency(s_1d)

        # get valid windows w.r.t. harmonics in frequency spectrum
        valid_windows = self._harmonics_analysis(s_1d, dominant_frequency, window_size, fft_factor, lp_freq_hz)

        # now we need to incorporate those windows that have already been discarded by the active signal detection
        gait_sequences_bool = np.copy(active_signal_mask)
        gait_sequences_bool[active_signal_mask] = valid_windows

        gait_sequences_start = gait_sequences_bool * range(len(gait_sequences_bool)) * overlap
        gait_sequences_start = gait_sequences_start[gait_sequences_start > 0]
        # concat subsequent gs
        gait_sequences_start_end = _gait_sequence_concat(sig_length, gait_sequences_start, window_size)

        gait_sequences_ = pd.DataFrame({"start": gait_sequences_start_end[:, 0], "end": gait_sequences_start_end[:, 1]})
        start_ = np.array(gait_sequences_["start"])
        end_ = np.array(gait_sequences_["end"])

        return gait_sequences_, start_, end_

    def _signal_extraction(self, data):
        """Extract the relevant signals and set required parameters from the data."""
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

        return s_3d_norm, s_1d, active_signal_th, fft_factor

    @staticmethod
    def _active_signal_detection(s_3d_norm, s_1d, active_signal_th):
        """Perform active signal detection based on 3d signal norm.

        Returns the s_1d reduced to the active windows and the boolean mask for the in-/active windows
        """
        # active signal detection
        # create boolean mask based on s_3d_norm
        active_signal_mask = np.mean(s_3d_norm, axis=1) > active_signal_th
        # only keep those rows with active signal
        s_1d = s_1d[active_signal_mask, :]
        return s_1d, active_signal_mask

    def _get_dominant_frequency(self, s_1d):
        """Compute the dominant frequency of each window using autocorrelation."""
        # dominant frequency via autocorrelation
        # set upper and lower motion band boundaries
        # TODO expose upper and lower bound as hidden parameters
        lower_bound = int(np.floor(self.sampling_rate_hz / 3))  # (=102.4 Hz / 3 Hz = upper bound of locomotor band)
        upper_bound = int(np.ceil(self.sampling_rate_hz / 0.5))  # (=102.4 Hz / 0.5 Hz = lower bound of locomotor band)
        # autocorr from 0-upper motion band
        auto_corr = _row_wise_autocorrelation(s_1d, upper_bound)
        # calculate dominant frequency in Hz
        dominant_frequency = (
            1 / (np.argmax(auto_corr[:, lower_bound:], axis=-1) + lower_bound).astype(float) * self.sampling_rate_hz
        )

        return dominant_frequency

    def _harmonics_analysis(self, s_1d, dominant_frequency, window_size, fft_factor, lp_freq_hz):
        """Analyze the frequency spectrum of s_1d regarding peaks at harmonics of the dominant frequency."""
        # determine harmonics candidates
        harmonics_candidates = np.outer(dominant_frequency, np.arange(2, 6))

        # compute the row-wise fft of the windowed signal and normalize it
        f_s_1d = np.abs(rfft(s_1d) / window_size) * fft_factor

        # Distance on the fft freq axis
        freq_axis_delta = self.sampling_rate_hz / 2 / f_s_1d.shape[1]

        # For efficient calculation, transform the row wise fft into an single 1D array
        f_s_1d_flat = f_s_1d.flatten()
        # Also flatten the harmonics array
        harmonics_flat = np.round(
            (harmonics_candidates / freq_axis_delta + (np.arange(f_s_1d.shape[0])[:, None] * f_s_1d.shape[1])).flatten()
        )

        # define size of window around candidates to look for peak. Allow 0.3 Hz of tolerance
        # TODO expose tolerance as hidden class attribute
        harmonic_window_half = int(np.floor(0.3 / freq_axis_delta))

        # TODO Edgecase: If close to a harmonic are 2 peaks, 1 with a high value, but peak prominence < threshold, and
        #  one which is a little bit lower, but with peak prominence > threshold. In this case martins method would
        #  have found the second peak and correctly concluded that the harmonic was found.
        #  With my method, I find the first and then conclude that the peak prominence is to low and discard it.
        closest_peaks = find_extrema_in_radius(f_s_1d_flat, harmonics_flat, harmonic_window_half, "max").astype(int)

        peak_prominence = peak_prominences(f_s_1d_flat, closest_peaks)[0].reshape(harmonics_candidates.shape)
        peak_heights = f_s_1d_flat[closest_peaks].reshape(harmonics_candidates.shape)

        # Apply thresholds
        # peaks should be higher than mean of f_s_1d
        min_peak_height = np.mean(f_s_1d[:, : np.floor(lp_freq_hz / freq_axis_delta).astype(int)], axis=1)
        # duplicate to match the shape of peak_heights
        min_peak_height = np.tile(min_peak_height, (peak_heights.shape[1], 1)).T

        harmonics_found = np.ones(harmonics_candidates.shape)
        harmonics_found[(peak_heights < min_peak_height) | (peak_prominence < self.peak_prominence)] = 0
        n_harmonics = harmonics_found.sum(axis=1)

        # find valid windows with n_harmonics > threshold
        n_harmonics_threshold = 2  # 2 as defined in the paper

        valid_windows = n_harmonics >= n_harmonics_threshold

        return valid_windows


# TODO consistent naming of variables in helper functions
def _butter_lowpass_filter(data, cutoff, sampling_rate_hz, order=4):
    """Create and apply butterworth lowpass filter."""
    nyq = 0.5 * sampling_rate_hz
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = lfilter(b, a, data)
    return y


@njit(nogil=True, parallel=True, cache=True)
def _row_wise_autocorrelation(array, lag_max):
    out = np.empty((array.shape[0], lag_max + 1))
    for tau in range(lag_max + 1):
        tmax = array.shape[1] - tau
        umax = array.shape[1] + tau
        out[:, tau] = (array[:, :tmax] * array[:, tau:umax]).sum(axis=1)
    return out


def _gait_sequence_concat(sig_length, gait_sequences_start, window_size):
    """Concat consecutive gait sequences to a single one."""
    # if there are no samples in the gait_sequences_start return the input
    if len(gait_sequences_start) == 0:
        gait_sequences_start_corrected = gait_sequences_start
    else:
        # empty list for result
        gait_sequences_start_corrected = []
        # first derivative of walking bout samples to get their relative distances
        gait_sequences_start_diff = np.diff(gait_sequences_start, axis=0)
        # compute those indices in the derivative where it is higher than the window size, these are the
        # non-consecutive bouts
        diff_jumps = np.where(gait_sequences_start_diff > window_size)[0]
        # split up the walking bout samples in the locations where they are not consecutive
        split_jumps = np.split(gait_sequences_start, diff_jumps + 1)
        # iterate over the single splits
        for jump in split_jumps:
            # start of the corrected walking bout is the first index of the jump
            start = jump[0]
            # length of the corrected walking bout is computed
            end = jump[-1] + window_size
            # if start+length exceeds the signal length correct the bout length
            if end > sig_length:
                end = sig_length
            # append list with start and length to the corrected walking bout samples
            gait_sequences_start_corrected.append([start, end])

    return np.array(gait_sequences_start_corrected)
