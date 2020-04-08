import numpy as np
import pytest
from numpy.testing import assert_array_equal

from gaitmap.utils.array_handling import (
    sliding_window_view,
    bool_array_to_start_end_array,
    split_array_at_nan,
    find_local_minima_below_threshold,
    find_minima_in_radius,
)


class TestSlidingWindow:
    """Test the function `sliding_window_view`."""

    def test_invalid_inputs_overlap(self):
        """Test if value error is raised correctly on invalid overlap input."""
        with pytest.raises(ValueError, match=r".* overlap .*"):
            sliding_window_view(np.arange(0, 10), window_length=4, overlap=4)

    def test_invalid_inputs_window_size(self):
        """Test if value error is raised correctly on invalid window length input."""
        with pytest.raises(ValueError, match=r".* window_length .*"):
            sliding_window_view(np.arange(0, 10), window_length=1, overlap=0)

    def test_view_of_array(selfs):
        """Test if output is actually just a different view onto the input data."""
        input_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        window_view = sliding_window_view(input_array, window_length=4, overlap=2)
        assert np.may_share_memory(input_array, window_view) == True

    def test_copy_of_array_with_padding(self):
        """Test if output a copy of input data."""
        input_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        window_view = sliding_window_view(input_array, window_length=4, overlap=2, nan_padding=True)
        assert np.may_share_memory(input_array, window_view) == False

    def test_nan_padding_of_type_nan(self):
        """Test if output a copy of input data."""
        input_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        window_view = sliding_window_view(input_array, window_length=4, overlap=2, nan_padding=True)

        import math

        assert math.isnan(window_view[-1][-1])

    def test_sliding_window_1D_without_without_padding(self):
        """Test windowed view is correct for 1D array without need for nan padding."""
        input_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        expected_output = np.array([[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9]])
        window_view = sliding_window_view(input_array, window_length=4, overlap=2)

        assert_array_equal(expected_output, window_view)

    def test_sliding_window_1D_with_padding(self):
        """Test windowed view is correct for 1D array with need for nan padding."""
        input_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        expected_output = np.array([[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9], [8, 9, 10, np.nan]])
        window_view = sliding_window_view(input_array, window_length=4, overlap=2, nan_padding=True)

        assert_array_equal(expected_output, window_view)

    def test_sliding_window_1D_without_padding(self):
        """Test windowed view is correct for 1D array with need for padding but padding disabled."""
        input_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        expected_output = np.array([[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9]])
        window_view = sliding_window_view(input_array, window_length=4, overlap=2, nan_padding=False)

        assert_array_equal(expected_output, window_view)

    def test_sliding_window_1D_asym_with_padding(self):
        """Test windowed view is correct for 1D array with need for nan padding and asymetrical overlap."""
        input_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        expected_output = np.array([[0, 1, 2, 3, 4, 5, 6], [5, 6, 7, 8, 9, np.nan, np.nan]])
        window_view = sliding_window_view(input_array, window_length=7, overlap=2, nan_padding=True)

        assert_array_equal(expected_output, window_view)

    def test_sliding_window_1D_asym_without_padding(self):
        """Test windowed view is correct for 1D array with need for nan padding but padding disabled and asymetrical
         overlap."""
        input_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        expected_output = np.array([0, 1, 2, 3, 4, 5, 6])
        window_view = sliding_window_view(input_array, window_length=7, overlap=2, nan_padding=False)

        assert_array_equal(expected_output, window_view)

    def test_sliding_window_1D_no_overlap_without_padding(self):
        """Test windowed view is correct for 1D array with no overlap and no padding."""
        input_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        expected_output = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        window_view = sliding_window_view(input_array, window_length=3, overlap=0)

        assert_array_equal(expected_output, window_view)

    def test_sliding_window_1D_no_overlap_with_padding(self):
        """Test windowed view is correct for 1D array with no overlap and need for padding."""
        input_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        expected_output = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, np.nan, np.nan]])
        window_view = sliding_window_view(input_array, window_length=4, overlap=0, nan_padding=True)

        assert_array_equal(expected_output, window_view)

    def test_sliding_window_1D_asym_overlap_with_padding(self):
        """Test windowed view is correct for 1D array with asym overlap need for padding."""
        input_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        expected_output = np.array([[0, 1, 2, 3, 4], [2, 3, 4, 5, 6], [4, 5, 6, 7, 8], [6, 7, 8, 9, np.nan]])
        window_view = sliding_window_view(input_array, window_length=5, overlap=3, nan_padding=True)

        assert_array_equal(expected_output, window_view)

    def test_sliding_window_3D_without_edge_case(self):
        """Test windowed view is correct for 3D array with sym overlap and no need for padding."""
        input_array = np.column_stack([np.arange(0, 10), np.arange(0, 10), np.arange(0, 10)])
        expected_output = np.array(
            [
                [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
                [[2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]],
                [[4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]],
                [[6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]],
            ]
        )
        window_view = sliding_window_view(input_array, window_length=4, overlap=2)

        assert_array_equal(expected_output, window_view)

    def test_sliding_window_3D_with_padding(self):
        """Test windowed view is correct for 3D array with sym overlap and need for padding."""
        input_array = np.column_stack([np.arange(0, 11), np.arange(0, 11), np.arange(0, 11)])
        expected_output = np.array(
            [
                [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
                [[2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]],
                [[4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]],
                [[6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]],
                [[8, 8, 8], [9, 9, 9], [10, 10, 10], [np.nan, np.nan, np.nan]],
            ]
        )
        window_view = sliding_window_view(input_array, window_length=4, overlap=2, nan_padding=True)

        assert_array_equal(expected_output, window_view)

    def test_sliding_window_3D_asym_with_padding(self):
        """Test windowed view is correct for 3D array with asym overlap and need for padding."""
        input_array = np.column_stack([np.arange(0, 12), np.arange(0, 12), np.arange(0, 12)])
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
        window_view = sliding_window_view(input_array, window_length=7, overlap=3, nan_padding=True)

        assert_array_equal(expected_output, window_view)

    def test_sliding_window_5D_without_padding(self):
        """Test windowed view is correct for high dimensional array with sym overlap and no need for padding."""
        input_array = np.column_stack(
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
        window_view = sliding_window_view(input_array, window_length=4, overlap=2)

        assert_array_equal(expected_output, window_view)


class TestBoolArrayToStartEndArray:
    """Test the function `sliding_window_view`."""

    def test_simple_input(self):
        input_array = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1])
        output_array = bool_array_to_start_end_array(input_array)
        expected_output = np.array([[2, 3], [6, 8]])
        assert_array_equal(expected_output, output_array)

    def test_invalid_inputs_overlap(self):
        """Test if value error is raised correctly on invalid input array."""
        with pytest.raises(ValueError, match=r".* boolean .*"):
            input_array = np.array([0, 0, 2, 2, 0, 0, 2, 2, 2])
            bool_array_to_start_end_array(input_array)

    def test_zeros_array(self):
        """Test zeros only input."""
        input_array = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        output_array = bool_array_to_start_end_array(input_array)
        assert output_array.size == 0

    def test_ones_array(self):
        """Test ones only input."""
        input_array = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        output_array = bool_array_to_start_end_array(input_array)
        expected_output = np.array([[0, 8]])
        assert_array_equal(expected_output, output_array)

    def test_edges_array(self):
        """Test correct handling of edges."""
        input_array = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1])
        output_array = bool_array_to_start_end_array(input_array)
        expected_output = np.array([[0, 2], [6, 8]])
        assert_array_equal(expected_output, output_array)

    def test_bool_value_array(self):
        """Test correct handling of boolean values."""
        input_array = np.array([True, True, True, False, False, False, True, True, True])
        output_array = bool_array_to_start_end_array(input_array)
        expected_output = np.array([[0, 2], [6, 8]])
        assert_array_equal(expected_output, output_array)

    def test_float_value_array(self):
        """Test correct handling of float values."""
        input_array = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        output_array = bool_array_to_start_end_array(input_array)
        expected_output = np.array([[0, 2], [6, 8]])
        assert_array_equal(expected_output, output_array)


class TestLocalMinimaBelowThreshold:
    @pytest.mark.parametrize(
        "data, result",
        [
            (np.array([np.nan]), []),
            (np.array([1, np.nan, 1]), [[1], [1]]),
            (np.array([1, 2, 3, np.nan, 4, 5]), [[1, 2, 3], [4, 5]]),
            (np.array([np.nan, 1, 2, np.nan, 3, 4, np.nan, 5, 6, np.nan]), [[1, 2], [3, 4], [5, 6]]),
            (
                np.array([np.nan, 1, 2, np.nan, np.nan, 3, 4, np.nan, np.nan, np.nan, 5, 6, np.nan]),
                [[1, 2], [3, 4], [5, 6]],
            ),
        ],
    )
    def test_split_array_simple(self, data, result):
        out = split_array_at_nan(data)
        assert len(out) == len(result)

        for a, b in zip(out, result):
            np.testing.assert_array_equal(a[1], np.array(b))

    @pytest.mark.parametrize(
        "data, threshold, results",
        [
            ([*np.ones(10), -1, -2, -1, *np.ones(10)], 0, [11]),
            (2 * [*np.ones(10), -1, -2, -1, *np.ones(10)], 0, [11, 34]),
            ([*np.ones(10), -1, -2, -1, *np.ones(10)], -3, []),
        ],
    )
    def test_find_extrema(self, data, threshold, results):
        out = find_local_minima_below_threshold(np.array(data), threshold)

        np.testing.assert_array_equal(out, results)


class TestFindMinRadius:
    def test_simple(self):
        data = np.array([0, 0, 0, -1, 0, 0, 0])  # min at 3
        radius = 1
        indices = np.array([2, 3, 4])  # All should find the minima
        out = find_minima_in_radius(data, indices, radius)
        assert_array_equal(out, [3, 3, 3])

    def test_multiple_matches(self):
        data = np.array([0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0])  # min at 3, 10
        radius = 2
        indices = np.arange(2, len(data) - 2)
        out = find_minima_in_radius(data, indices, radius)
        # 2 - 5 should see the first minimum, 6, 7 see no minimum, 8-11 see second minimum
        assert_array_equal(out, [3, 3, 3, 3, 4, 5, 10, 10, 10, 10])

    def test_edge_case_end(self):
        data = np.array([0, 0, 0, 0, 0, -1, 0])  # min at 5
        radius = 2
        indices = np.array([4, 5])  # 5 overlap with end
        out = find_minima_in_radius(data, indices, radius)
        assert_array_equal(out, [5, 5])

    def test_edge_case_start(self):
        data = np.array([0, -1, 0, 0, 0, 0, 0])  # min at 1
        radius = 2
        indices = np.array([1, 2])  # 1 overlap with start
        out = find_minima_in_radius(data, indices, radius)
        assert_array_equal(out, [1, 1])

    def test_full_dummy(self):
        """As there is no minimum, every index should return the start of the window."""
        length = 10
        data = np.zeros(10)
        indices = np.arange(10)
        radius = 2
        out = find_minima_in_radius(data, indices, radius)
        assert_array_equal(out[radius:], indices[radius:] - radius)
        assert_array_equal(out[:radius], np.zeros(radius))
