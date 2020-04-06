import numpy as np
import pytest
from numpy.testing import assert_array_equal

from gaitmap.utils.static_moment_detection import find_static_samples, find_static_sequences


class TestFindStaticSamples:
    """Test the function `sliding_window_view`."""

    def test_invalid_input_dimension(self):
        """Test if value error is raised correctly on invalid input dimensions."""
        test_input = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
        with pytest.raises(ValueError, match=r".* dimensions.*"):
            find_static_samples(test_input, window_length=4, overlap=3, inactive_signal_th=0, metric="maximum")

    def test_invalid_input_metric(self):
        """Test if value error is raised correctly on invalid input dimensions."""
        test_input = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
        test_input = np.column_stack([test_input, test_input, test_input])
        with pytest.raises(ValueError, match=r".*Invalid metric passed!.*"):
            find_static_samples(
                test_input, window_length=4, overlap=3, inactive_signal_th=0, metric="some_invalid_metric"
            )

    def test_max_overlap_metric_max_w4(self):
        """Test binary input data on max metric with window size 4."""
        test_input = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
        test_input = np.column_stack([test_input, test_input, test_input])
        expected_output = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0])

        window_length = 4
        test_output = find_static_samples(
            test_input, window_length=window_length, overlap=window_length - 1, inactive_signal_th=0, metric="maximum"
        )
        assert_array_equal(test_output, expected_output)

    def test_max_overlap_metric_max_w3(self):
        """Test binary input data on max metric with window size 3."""
        test_input = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
        test_input = np.column_stack([test_input, test_input, test_input])
        expected_output = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0])

        window_length = 3
        test_output = find_static_samples(
            test_input, window_length=window_length, overlap=window_length - 1, inactive_signal_th=0, metric="maximum"
        )
        assert_array_equal(test_output, expected_output)

    def test_max_overlap_metric_max_w6(self):
        """Test binary input data on max metric with window size 6."""
        test_input = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
        test_input = np.column_stack([test_input, test_input, test_input])
        expected_output = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        window_length = 6
        test_output = find_static_samples(
            test_input, window_length=window_length, overlap=window_length - 1, inactive_signal_th=0, metric="maximum"
        )
        assert_array_equal(test_output, expected_output)

    def test_max_overlap_mean_w3_with_noise(self):
        """Test binary input data on mean metric with window size 4 after adding a bit of noise."""
        test_input = data = np.array([0, 0.1, 0, 0, 0.1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0.1, 0, 0, 1, 1])
        test_input = np.column_stack([test_input, test_input, test_input])
        expected_output = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0])

        window_length = 3
        test_output = find_static_samples(
            test_input, window_length=window_length, overlap=window_length - 1, inactive_signal_th=0.1, metric="mean"
        )
        assert_array_equal(test_output, expected_output)

    def test_max_overlap_max_w3_with_noise(self):
        """Test binary input data on max metric with window size 4 after adding a bit of noise."""
        test_input = data = np.array([0, 0.1, 0, 0, 0.1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0.1, 0, 0, 1, 1])
        test_input = np.column_stack([test_input, test_input, test_input])
        expected_output = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        window_length = 3
        test_output = find_static_samples(
            test_input, window_length=window_length, overlap=window_length - 1, inactive_signal_th=0.1, metric="maximum"
        )
        assert_array_equal(test_output, expected_output)


class TestFindStaticSequences:
    """Test the function `sliding_window_view`."""

    def test_max_overlap_metric_max_w4(self):
        """Test binary input data on max metric with window size 4."""
        test_input = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
        test_input = np.column_stack([test_input, test_input, test_input])
        expected_output = np.array([[0, 5], [16, 20]])

        window_length = 4
        test_output = find_static_sequences(
            test_input, window_length=window_length, overlap=window_length - 1, inactive_signal_th=0, metric="maximum"
        )
        assert_array_equal(test_output, expected_output)

    def test_max_overlap_metric_mean_w4(self):
        """Test binary input data on max metric with window size 4."""
        test_input = np.array([0, 0, 0.1, 0, 0.1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0.1, 0, 0, 1, 1])
        test_input = np.column_stack([test_input, test_input, test_input])
        expected_output = np.array([[0, 5], [16, 20]])

        window_length = 4
        test_output = find_static_sequences(
            test_input, window_length=window_length, overlap=window_length - 1, inactive_signal_th=0.1, metric="mean"
        )
        assert_array_equal(test_output, expected_output)
