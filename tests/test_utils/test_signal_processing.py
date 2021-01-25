import numpy as np

from gaitmap.utils.signal_processing import row_wise_autocorrelation


def test_row_wise_autocorrelation():
    """Test if the manually implemented row wise autocorrelation function produces similar results than the numpy
    implementation"""

    # create a test signal
    sampling_rate = 204.8
    samples = 100
    t = np.arange(samples) / sampling_rate
    freq = 1
    test_signal_sin = np.sin(2 * np.pi * freq * t)
    test_signal_cos = np.cos(2 * np.pi * freq * t)
    test_signal_array = np.array([test_signal_sin, test_signal_cos])

    max_lag = 99
    autocorr_row_wise = row_wise_autocorrelation(test_signal_array, max_lag)

    for idx, test_signal in enumerate([test_signal_sin, test_signal_cos]):
        autocorr_1d = np.correlate(test_signal, test_signal, mode="full")
        autocorr_1d = autocorr_1d[autocorr_1d.size // 2 :][: max_lag + 1]

        # for some reason the values are not exactly equal but difference is smaller than eps
        np.testing.assert_allclose(autocorr_1d, autocorr_row_wise[idx])
