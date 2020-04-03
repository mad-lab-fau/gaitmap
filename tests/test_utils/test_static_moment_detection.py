import numpy as np
import pytest
from numpy.testing import assert_array_equal

from gaitmap.utils.static_moment_detection import sliding_window_view


class TestSlidingWindow:
    """Test the function `sliding_window`."""

    def test_invalid_inputs_overlap(self):
        with pytest.raises(ValueError, match=r".* overlap .*"):
            sliding_window_view(np.arange(0, 10), window_length=4, window_overlap=4)

    def test_invalid_inputs_window_size(self):
        with pytest.raises(ValueError, match=r".* window_length .*"):
            sliding_window_view(np.arange(0, 10), window_length=1, window_overlap=0)

    def test_sliding_window_1D_without_edge_case(self):
        input = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        expected_output = np.array([[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9]])
        window_view = sliding_window_view(input, window_length=4, window_overlap=2)

        assert_array_equal(expected_output, window_view)

    def test_sliding_window_1D_with_edge_case(self):
        input = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        expected_output = np.array([[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9], [8, 9, 10, np.nan]])
        window_view = sliding_window_view(input, window_length=4, window_overlap=2)

        assert_array_equal(expected_output, window_view)

    def test_sliding_window_1D_no_overlap_without_edge_case(self):
        input = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        expected_output = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        window_view = sliding_window_view(input, window_length=3, window_overlap=0)

        assert_array_equal(expected_output, window_view)

    def test_sliding_window_1D_no_overlap_with_edge_case(self):
        input = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        expected_output = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, np.nan, np.nan]])
        window_view = sliding_window_view(input, window_length=4, window_overlap=0)

        assert_array_equal(expected_output, window_view)
