"""The gait sequence detection algorithm by Rampp et al. 2020."""
import itertools
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
    sensor_channel_config
        The sensor channel or sensor that should be used for the gait sequence detection. Must be a str. If the
        algorithm should be applied to a single sensor axis the string must be one of the entries in either
         :obj:`~gaitmap.utils.consts.BF_ACC` or :obj:`~gaitmap.utils.consts.BF_GYR`. In order to perform the analysis
         on the norm of the acc / the gyr signal, you need to pass "acc" or "gyr". Default value is `gyr_ml` as by the
         publication this showed the best results.
    peak_prominence
        The threshold for the peak prominence that each harmonics peak must provide. This relates to the
        `peak_prominence` provided by :func:`~scipy.signal.peak_prominences`. The frequency spectrum of the sensor
        signal will be investigated at harmonics of the dominant frequency regarding peaks that have a
        `peak_prominence` higher than the specified value. The lower the value, the more inclusive is the algorithm,
        meaning that the requirements of a signal to be detected as gait are lower. The higher the value the more
        specific is the algorithm to include a signal as gait.
    window_size_s
        The algorithm works on a window basis. The `window_size_s` specified the length of this window in seconds.
        Default value is 10 s. The windowing in this algorithm works with a 50% overlap of subsequent windows.
    active_signal_threshold
        Before the actual FFT analysis, the algorithm compares the mean signal norm within each window against the
        `active_signal_threshold` in order to reject windows with rest right from the beginning. Default value is
        according to the publication is 50 deg / s for `sensor_channel_config` settings including the gyroscope or
        0.2 g for `sensor_channel_config` settings including the accelerometer. For higher thresholds the signal in
        the window must show more activity to be passed to the frequency analysis.
    locomotion_band
        The `locomotion_band` defines the region within the frequency spectrum of each window to identify the dominant
        frequency. According to literature [  2]_ typical signals measured during locomotion have their dominant
        frequency within a locomotion band of 0.5 - 3 Hz. This is set as the default value. For very slow or very
        fast gait these borders might have to be adapted.
    harmonic_tolerance_hz
        After identifying the dominant frequency of a window, the algorithm checks for peaks in the frequency
        spectrum that should appear at harmonics of the dominant frequency. The `harmonic_tolerance_hz` sets the
        tolerance margin for the harmonics. Default value is 0.3 Hz. This means if a harmonic peak is detected up
        to 0.3 Hz above or 0.3 Hz below the actual harmonic it is still accepted.
    merge_gait_sequences_from_sensors
        The algorithm processes the data from each sensor individually. Therefore, different gait sequences for
        example for left and right foot, may occur. If `merge_gait_sequences_from_sensors` is set to True,
        the gait sequences from all sensors will be merged by applying a logical `OR` to the single results. This is
        only allowed for synchronized sensor data.

    Attributes
    ----------
    TODO Is there going to be datatype 'gait_sequence_list'? Can this be a subtype of a stride_list?
    gait_sequences_ : A pandas.DataFrame or dictionary with pandas.DataFrames
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

    Examples
    --------
    Find sequences of gait in sensor signal

    >>> gsd = UllrichGaitSequenceDetection()
    >>> gsd = gsd.detect(data, 204.8)
    >>> gsd.gait_sequences_
        gs_id  start     end
    0   0       1024      3072
    1   1       4096      8192
    ...

    Notes
    -----
    TODO: Add additional details about the algorithm for gait sequence detection

    .. [1] M. Ullrich, A. Küderle, J. Hannink, S. Del Din, H. Gassner, F. Marxreiter, J. Klucken, B.M.
        Eskofier, F. Kluge, Detection of Gait From Continuous Inertial Sensor Data Using Harmonic
        Frequencies, IEEE Journal of Biomedical and Health Informatics. (2020) 1–1.
        https://doi.org/10.1109/JBHI.2020.2975361.

    .. [2] Iluz, T., Gazit, E., Herman, T. et al. Automated detection of missteps during community ambulation in
        patients with Parkinson’s disease: a new approach for quantifying fall risk in the community setting.
        J NeuroEngineering Rehabil 11, 48 (2014). https://doi.org/10.1186/1743-0003-11-48


    """

    sensor_channel_config: str
    peak_prominence: float
    window_size_s: float

    start_: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
    end_: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
    gait_sequences_: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]

    data: Dataset
    sampling_rate_hz: float

    def __init__(
        self,
        sensor_channel_config: str = "gyr_ml",
        peak_prominence: float = 17.0,
        window_size_s: float = 10,
        active_signal_threshold: float = None,
        locomotion_band: Tuple[float, float] = (0.5, 3),
        harmonic_tolerance_hz: float = 0.3,
        merge_gait_sequences_from_sensors: bool = False,
    ):
        self.sensor_channel_config = sensor_channel_config
        self.peak_prominence = peak_prominence
        self.window_size_s = window_size_s
        self.active_signal_threshold = active_signal_threshold
        self.locomotion_band = locomotion_band
        self.harmonic_tolerance_hz = harmonic_tolerance_hz
        self.merge_gait_sequences_from_sensors = merge_gait_sequences_from_sensors

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

        """
        self.data = data
        self.sampling_rate_hz = sampling_rate_hz

        # TODO implement merging of gait sequences for synced data
        if self.merge_gait_sequences_from_sensors and is_multi_sensor_dataset(data) and isinstance(data, pd.DataFrame):
            raise NotImplementedError("Merging of gait sequences from several sensors is not yet supported.")

        self._assert_input_data(data)

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
        s_3d, s_1d, active_signal_th, fft_factor = self._signal_extraction(data)

        # sig_length is required later for the concatenation of gait sequences
        sig_length = len(s_1d)

        # lowpass filter the signal: this is now happening before the windowing and thus before the active signal
        # detection compared to JBHI version
        lp_freq_hz = 6  # 6 Hz as harmonics are supposed to occur in lower frequencies
        s_1d = _butter_lowpass_filter(s_1d, lp_freq_hz, self.sampling_rate_hz)

        # sliding windows
        overlap = int(window_size / 2)
        if window_size > len(s_3d):
            raise ValueError("The selected window size is larger than the actual signal.")
        s_3d = sliding_window_view(s_3d, window_size, overlap)
        # subtract mean per window
        s_3d = s_3d - np.mean(s_3d, axis=1)[:, np.newaxis, :]
        # compute norm per window
        s_3d_norm = norm(s_3d, axis=2)

        s_1d = sliding_window_view(s_1d, window_size, overlap)
        # subtract mean per window
        s_1d = s_1d - np.mean(s_1d, axis=1)[:, np.newaxis]

        # active signal detection
        s_1d, active_signal_mask = self._active_signal_detection(s_3d_norm, s_1d, active_signal_th)

        if s_1d.size == 0:
            gait_sequences_ = pd.DataFrame(columns=["gs_id", "start", "end"])
            start_ = np.array(gait_sequences_["start"])
            end_ = np.array(gait_sequences_["end"])

            return gait_sequences_, start_, end_

        # dominant frequency via autocorrelation
        dominant_frequency = self._get_dominant_frequency(s_1d)

        # get valid windows w.r.t. harmonics in frequency spectrum
        valid_windows = self._harmonics_analysis(s_1d, dominant_frequency, window_size, fft_factor, lp_freq_hz)

        # now we need to incorporate those windows that have already been discarded by the active signal detection
        gait_sequences_bool = np.copy(active_signal_mask)
        gait_sequences_bool[active_signal_mask] = valid_windows

        # we need to distinguish between even or odd windows sizes to compute the correct start samples
        if window_size % 2 == 0:
            gait_sequences_start = (np.arange(len(gait_sequences_bool)) * overlap)[gait_sequences_bool]
        else:
            gait_sequences_start = (np.arange(len(gait_sequences_bool)) * (overlap + 1))[gait_sequences_bool]
        # concat subsequent gs
        gait_sequences_start_end = _gait_sequence_concat(sig_length, gait_sequences_start, window_size)

        gait_sequences_ = pd.DataFrame({"start": gait_sequences_start_end[:, 0], "end": gait_sequences_start_end[:, 1]})

        # add a column for the gs_id
        gait_sequences_ = gait_sequences_.reset_index().rename(columns={"index": "gs_id"})
        start_ = np.array(gait_sequences_["start"])
        end_ = np.array(gait_sequences_["end"])

        return gait_sequences_, start_, end_

    def _signal_extraction(self, data):
        """Extract the relevant signals and set required parameters from the data."""
        # define 3d signal to analyze for active signal and further parameters
        active_signal_th = self.active_signal_threshold

        # define the 3d signal, fft factor and active signal th
        if "acc" in self.sensor_channel_config:
            s_3d = np.array(data[BF_ACC])
            # scaling factor for the FFT result to get comparable numerical range for acc or gyr configurations
            fft_factor = 100
            # TODO is it fine to only check for None and finally set this value here?
            if active_signal_th is None:
                active_signal_th = 0.2
        else:
            s_3d = np.array(data[BF_GYR])
            fft_factor = 1
            if active_signal_th is None:
                active_signal_th = 50

        # determine the 1d signal that is going to be used for the frequency analysis
        if self.sensor_channel_config in list(itertools.chain(BF_ACC, BF_GYR)):
            s_1d = np.array(data[self.sensor_channel_config])
        else:
            s_1d = norm(s_3d, axis=1)

        return s_3d, s_1d, active_signal_th, fft_factor

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
        # set upper and lower motion band boundaries
        # (sampling_rate / locomotion_band_upper = lower bound of autocorrelation)
        lower_bound = int(np.floor(self.sampling_rate_hz / self.locomotion_band[1]))
        # (sampling_rate / locomotion_band_lower = upper bound of autocorrelation)
        upper_bound = int(np.ceil(self.sampling_rate_hz / self.locomotion_band[0]))
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
        # removing the last sample of the fft to be consistent with JBHI implementation
        f_s_1d = f_s_1d[:, :-1]

        # Distance on the fft freq axis
        freq_axis_delta = self.sampling_rate_hz / 2 / f_s_1d.shape[1]

        # For efficient calculation, transform the row wise fft into an single 1D array
        f_s_1d_flat = f_s_1d.flatten()
        # Also flatten the harmonics array
        harmonics_flat = np.round(
            (harmonics_candidates / freq_axis_delta + (np.arange(f_s_1d.shape[0])[:, None] * f_s_1d.shape[1])).flatten()
        )

        if self.harmonic_tolerance_hz < freq_axis_delta:
            raise ValueError("Value for harmonic_tolerance_hz too small. Must be > {} ".format(freq_axis_delta))
        # define size of window around candidates to look for peak. Allow per default 0.3 Hz of tolerance
        harmonic_window_half = int(np.floor(self.harmonic_tolerance_hz / freq_axis_delta))

        # INFO Edgecase: If close to a harmonic are 2 peaks, 1 with a high value, but peak prominence < threshold, and
        #  one which is a little bit lower, but with peak prominence > threshold. In this case the original
        #  implementation would have found the second peak and correctly concluded that the harmonic was found.
        #  With this implementation method, we find the first and then conclude that the peak prominence is too low and
        #  discard it.
        closest_peaks = find_extrema_in_radius(f_s_1d_flat, harmonics_flat, harmonic_window_half, "max").astype(int)

        peak_prominence = peak_prominences(f_s_1d_flat, closest_peaks)[0].reshape(harmonics_candidates.shape)
        peak_heights = f_s_1d_flat[closest_peaks].reshape(harmonics_candidates.shape)

        # Apply thresholds
        # peaks should be higher than mean of f_s_1d in the area <= lp_freq_hz. Have to add a + 1 to include the limit
        min_peak_height = np.mean(f_s_1d[:, : np.floor(lp_freq_hz / freq_axis_delta).astype(int) + 1], axis=1)
        # duplicate to match the shape of peak_heights
        min_peak_height = np.tile(min_peak_height, (peak_heights.shape[1], 1)).T

        harmonics_found = np.ones(harmonics_candidates.shape)
        harmonics_found[(peak_heights < min_peak_height) | (peak_prominence < self.peak_prominence)] = 0
        n_harmonics = harmonics_found.sum(axis=1)

        # find valid windows with n_harmonics > threshold
        n_harmonics_threshold = 2  # 2 as defined in the paper

        valid_windows = n_harmonics >= n_harmonics_threshold

        return valid_windows

    def _assert_input_data(self, data):
        if (
            self.merge_gait_sequences_from_sensors
            and is_multi_sensor_dataset(data)
            and not isinstance(data, pd.DataFrame)
        ):
            raise ValueError("Merging of data set is only possible for synchronized data sets.")

        # check for correct input value for sensor_channel_config
        if isinstance(self.sensor_channel_config, str) and self.sensor_channel_config not in list(
            itertools.chain(BF_ACC, BF_GYR, ["gyr", "acc"])
        ):
            raise ValueError(
                "The sensor_channel_config str you have passed is invalid. If you pass a str it must be one of the "
                "entries of either BF_ACC or BF_GYR or just gyr or acc in order to apply the norm of the respective "
                "sensor."
            )

        if not isinstance(self.sensor_channel_config, str):
            raise ValueError("Sensor_channel_config must be a str.")

        # check locomotion band
        if len(self.locomotion_band) != 2:
            raise ValueError("The tuple for the locomotion band must contain exactly two values.")
        if self.locomotion_band[0] >= self.locomotion_band[1]:
            raise ValueError("The first entry of the locomotion band must be smaller than the second value.")
        if self.locomotion_band[0] <= 0:
            raise ValueError("The first entry of the locomotion band must be larger than 0.")

        # check if 5 * upper freq range limit is close to nyquist freq (allow for a difference > 5 Hz). This would
        # cause edge cases for the flattened fft peak detection later on.
        if (self.sampling_rate_hz / 2) - (5 * self.locomotion_band[1]) < 5:
            raise ValueError(
                "The upper limit of the locomotion band is too close to the Nyquist frequency of the "
                "signal. It should be smaller than 5 Hz."
            )


# TODO move this to general utils
def _butter_lowpass_filter(data, cutoff, sampling_rate_hz, order=4):
    """Create and apply butterworth lowpass filter."""
    nyq = 0.5 * sampling_rate_hz
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = lfilter(b, a, data)
    return y


# TODO move this to general utils
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
