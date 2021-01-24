import numpy as np
import pytest
from numpy.testing import assert_array_equal

from gaitmap.utils.array_handling import bool_array_to_start_end_array
from gaitmap.utils.static_moment_detection import (
    find_static_samples,
    find_static_sequences,
    find_first_static_window_multi_sensor,
)


class TestFindStaticSamples:
    """Test the function `sliding_window_view`."""

    def test_invalid_input_dimension_default_overlap(self):
        """Test if value error is raised correctly on invalid input dimensions."""
        test_input = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
        with pytest.raises(ValueError, match=r".* dimensions.*"):
            find_static_samples(test_input, window_length=4, inactive_signal_th=0, metric="maximum")

    def test_invalid_input_metric_default_overlap(self):
        """Test if value error is raised correctly on invalid input dimensions."""
        test_input = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
        test_input = np.column_stack([test_input, test_input, test_input])
        with pytest.raises(ValueError, match=r".*Invalid metric passed!.*"):
            find_static_samples(
                test_input, window_length=4, overlap=3, inactive_signal_th=0, metric="some_invalid_metric"
            )

    def test_invalid_window_length(self):
        """Test if value error is raised correctly on invalid input dimensions."""
        test_input = np.array([0, 0, 0, 0, 0, 0])
        test_input = np.column_stack([test_input, test_input, test_input])
        with pytest.raises(ValueError, match=r".*Invalid window length*"):
            find_static_samples(test_input, window_length=10, overlap=3, inactive_signal_th=0, metric="maximum")

    def test_single_window_fit(self):
        """Test input where only a single window length fits within input signal."""
        test_input = np.array([0, 0, 0, 0, 0, 1, 1, 1])
        test_input = np.column_stack([test_input, test_input, test_input])
        expected_output = np.array([1, 1, 1, 1, 1, 0, 0, 0])
        window_length = 5
        test_output = find_static_samples(
            test_input, window_length=window_length, inactive_signal_th=0, metric="maximum", overlap=1
        )
        assert_array_equal(test_output, expected_output)

    def test_max_overlap_metric_max_w4_default_overlap(self):
        """Test binary input data on max metric with window size 4."""
        test_input = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
        test_input = np.column_stack([test_input, test_input, test_input])
        expected_output = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0])

        window_length = 4
        test_output = find_static_samples(
            test_input, window_length=window_length, inactive_signal_th=0, metric="maximum"
        )
        assert_array_equal(test_output, expected_output)

    def test_max_overlap_metric_max_w3(self):
        """Test binary input data on max metric with window size 3."""
        test_input = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
        test_input = np.column_stack([test_input, test_input, test_input])
        expected_output = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0])

        window_length = 3
        test_output = find_static_samples(
            test_input, window_length=window_length, inactive_signal_th=0, metric="maximum", overlap=window_length - 1
        )
        assert_array_equal(test_output, expected_output)

    def test_max_overlap_metric_max_w6_default_overlap(self):
        """Test binary input data on max metric with window size 6."""
        test_input = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
        test_input = np.column_stack([test_input, test_input, test_input])
        expected_output = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        window_length = 6
        test_output = find_static_samples(
            test_input, window_length=window_length, inactive_signal_th=0, metric="maximum"
        )
        assert_array_equal(test_output, expected_output)

    def test_max_overlap_mean_w3_with_noise_default_overlap(self):
        """Test binary input data on mean metric with window size 4 after adding a bit of noise."""
        test_input = data = np.array([0, 0.1, 0, 0, 0.1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0.1, 0, 0, 1, 1])
        test_input = np.column_stack([test_input, test_input, test_input])
        expected_output = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0])

        window_length = 3
        test_output = find_static_samples(
            test_input, window_length=window_length, inactive_signal_th=0.1, metric="mean"
        )
        assert_array_equal(test_output, expected_output)

    def test_max_overlap_max_w3_with_noise(self):
        """Test binary input data on max metric with window size 4 after adding a bit of noise."""
        test_input = data = np.array([0, 0.1, 0, 0, 0.1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0.1, 0, 0, 1, 1])
        test_input = np.column_stack([test_input, test_input, test_input])
        expected_output = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        window_length = 3
        test_output = find_static_samples(
            test_input, window_length=window_length, inactive_signal_th=0.1, metric="maximum", overlap=window_length - 1
        )
        assert_array_equal(test_output, expected_output)


class TestFindStaticSequences:
    """Test the function `sliding_window_view`."""

    def test_max_overlap_metric_max_w4_default_overlap(self):
        """Test binary input data on max metric with window size 4."""
        test_input = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
        test_input = np.column_stack([test_input, test_input, test_input])
        expected_output = np.array([[0, 6], [16, 21]])

        window_length = 4
        test_output = find_static_sequences(
            test_input, window_length=window_length, inactive_signal_th=0, metric="maximum"
        )
        assert_array_equal(test_output, expected_output)

    def test_max_overlap_metric_mean_w4_default_overlap(self):
        """Test binary input data on max metric with window size 4."""
        test_input = np.array([0, 0, 0.1, 0, 0.1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0.1, 0, 0, 1, 1])
        test_input = np.column_stack([test_input, test_input, test_input])
        expected_output = np.array([[0, 6], [16, 21]])

        window_length = 4
        test_output = find_static_sequences(
            test_input, window_length=window_length, inactive_signal_th=0.1, metric="mean"
        )
        assert_array_equal(test_output, expected_output)


class TestFirstStaticWindowsMultiSensor:
    def test_basic_single_sensor(self):
        test_input = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0])
        test_input = np.column_stack([test_input, test_input, test_input])[:, None, :]

        window_length = 4
        test_output = find_first_static_window_multi_sensor(
            test_input, window_length=window_length, inactive_signal_th=0, metric="maximum"
        )
        assert_array_equal(test_output, (6, 10))

    def test_basic_multi_sensor(self):
        test_input = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        test_input_2 = np.array([1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
        test_input = np.column_stack([test_input, test_input, test_input])
        test_input_2 = np.column_stack([test_input_2, test_input_2, test_input_2])

        window_length = 4
        test_output = find_first_static_window_multi_sensor(
            [test_input, test_input_2], window_length=window_length, inactive_signal_th=0, metric="maximum"
        )
        assert_array_equal(test_output, (8, 12))

    def test_invalid_shape_sub_array(self):
        test_input = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        test_input_2 = np.array([1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0])

        window_length = 4
        with pytest.raises(ValueError) as e:
            find_first_static_window_multi_sensor(
                [test_input, test_input_2], window_length=window_length, inactive_signal_th=0, metric="maximum"
            )
        assert "2D" in str(e)

    def test_invalid_shape_np_array(self):
        test_input = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        test_input = np.column_stack([test_input, test_input, test_input])

        window_length = 4
        with pytest.raises(ValueError) as e:
            find_first_static_window_multi_sensor(
                test_input, window_length=window_length, inactive_signal_th=0, metric="maximum"
            )
        assert "3D" in str(e)

    def test_invalid_metric(self):
        test_input = np.array([1, 1, 1, 1, 1, 1])
        test_input = np.column_stack([test_input, test_input, test_input])[:, None, :]

        window_length = 4

        with pytest.raises(ValueError) as e:
            find_first_static_window_multi_sensor(
                test_input, window_length=window_length, inactive_signal_th=0, metric="invalid metric"
            )

        assert "metric" in str(e)

    def test_no_static_window(self):
        test_input = np.array([1, 1, 1, 1, 1, 1])
        test_input = np.column_stack([test_input, test_input, test_input])[:, None, :]

        window_length = 4

        with pytest.raises(ValueError) as e:
            find_first_static_window_multi_sensor(
                test_input, window_length=window_length, inactive_signal_th=0, metric="maximum"
            )

        assert "No static" in str(e)
