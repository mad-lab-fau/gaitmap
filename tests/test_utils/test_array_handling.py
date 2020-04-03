import numpy as np
import pytest
from numpy.testing import assert_array_equal

from gaitmap.utils.array_handling import sliding_window_view


class TestSlidingWindow:
    """Test the function `sliding_window`."""

    def test_invalid_inputs_overlap(self):
        with pytest.raises(ValueError, match=r".* overlap .*"):
            sliding_window_view(np.arange(0, 10), window_length=4, overlap=4)

    def test_invalid_inputs_window_size(self):
        with pytest.raises(ValueError, match=r".* window_length .*"):
            sliding_window_view(np.arange(0, 10), window_length=1, overlap=0)

    def test_sliding_window_1D_without_edge_case(self):
        input = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        expected_output = np.array([[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9]])
        window_view = sliding_window_view(input, window_length=4, overlap=2)

        assert_array_equal(expected_output, window_view)

    def test_sliding_window_1D_with_edge_case(self):
        input = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        expected_output = np.array([[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9], [8, 9, 10, np.nan]])
        window_view = sliding_window_view(input, window_length=4, overlap=2)

        assert_array_equal(expected_output, window_view)

    def test_sliding_window_1D_with_edge_case2(self):
        input = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        expected_output = np.array([[0, 1, 2, 3, 4, 5, 6], [5, 6, 7, 8, 9, np.nan, np.nan]])
        window_view = sliding_window_view(input, window_length=7, overlap=2)

        assert_array_equal(expected_output, window_view)

    def test_sliding_window_1D_no_overlap_without_edge_case(self):
        input = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        expected_output = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        window_view = sliding_window_view(input, window_length=3, overlap=0)

        assert_array_equal(expected_output, window_view)

    def test_sliding_window_1D_no_overlap_with_edge_case(self):
        input = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        expected_output = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, np.nan, np.nan]])
        window_view = sliding_window_view(input, window_length=4, overlap=0)

        assert_array_equal(expected_output, window_view)

    def test_sliding_window_1D_asym_overlap_with_edge_case(self):
        input = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        expected_output = np.array([[0, 1, 2, 3, 4], [2, 3, 4, 5, 6], [4, 5, 6, 7, 8], [6, 7, 8, 9, np.nan]])
        window_view = sliding_window_view(input, window_length=5, overlap=3)

        assert_array_equal(expected_output, window_view)

    def test_sliding_window_3D_without_edge_case(self):
        window_length = 4
        overlap = 2
        input = np.column_stack([np.arange(0, 10), np.arange(0, 10), np.arange(0, 10)])
        expected_output = np.array(
            [
                [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
                [[2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]],
                [[4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]],
                [[6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]],
            ]
        )
        window_view = sliding_window_view(input, window_length, overlap)

        assert_array_equal(expected_output, window_view)

    def test_sliding_window_3D_with_edge_case(self):
        window_length = 4
        overlap = 2
        input = np.column_stack([np.arange(0, 11), np.arange(0, 11), np.arange(0, 11)])
        expected_output = np.array(
            [
                [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
                [[2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]],
                [[4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]],
                [[6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]],
                [[8, 8, 8], [9, 9, 9], [10, 10, 10], [np.nan, np.nan, np.nan]],
            ]
        )
        window_view = sliding_window_view(input, window_length, overlap)

        assert_array_equal(expected_output, window_view)

    def test_sliding_window_3D_with_edge_case2(self):
        window_length = 7
        overlap = 3
        input = np.column_stack([np.arange(0, 12), np.arange(0, 12), np.arange(0, 12)])
        expected_output = np.array(
            [
                [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6]],
                [[4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9], [10, 10, 10]],
                [
                    [8, 8, 8],
                    [9, 9, 9],
                    [10, 10, 10],
                    [11, 11, 11],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                ],
            ]
        )
        window_view = sliding_window_view(input, window_length, overlap)

        assert_array_equal(expected_output, window_view)

    def test_sliding_window_5D_without_edge_case(self):
        window_length = 4
        overlap = 2
        input = np.column_stack(
            [np.arange(0, 10), np.arange(0, 10), np.arange(0, 10), np.arange(0, 10), np.arange(0, 10)]
        )
        expected_output = np.array(
            [
                [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]],
                [[2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4], [5, 5, 5, 5, 5]],
                [[4, 4, 4, 4, 4], [5, 5, 5, 5, 5], [6, 6, 6, 6, 6], [7, 7, 7, 7, 7]],
                [[6, 6, 6, 6, 6], [7, 7, 7, 7, 7], [8, 8, 8, 8, 8], [9, 9, 9, 9, 9]],
            ]
        )
        window_view = sliding_window_view(input, window_length, overlap)

        assert_array_equal(expected_output, window_view)
