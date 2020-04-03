import pytest
import numpy as np
from numpy.linalg import norm
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_almost_equal

from gaitmap.utils.vector_math import (
    find_random_orthogonal,
    find_orthogonal,
    find_unsigned_3d_angle,
    normalize,
    is_almost_parallel_or_antiprallel,
)


class TestIsAlmostParallelOrAntiprallel:
    """Test the function `is_almost_parallel_or_antiprallel`."""

    @pytest.mark.parametrize(
        "v1, v2, result",
        [
            ([1, 0, 0], [1, 1, 0], False),
            ([1, 0, 0], [1, 0, 0], True),
            ([1, 0, 0], [0, 1, 0], False),
            ([1, 0, 1], [2, 0, 2], True),
            ([0, -1, 0], [0, 2, 0], True),
            ([0, 0, 2], [0, 0, 2], True),
            ([1, -1, 0], [1, -1, 0], True),
            ([0, -1, 2], [0, -1, 2], True),
        ],
    )
    def test_is_almost_parallel_or_antiprallel_single_vector(self, v1, v2, result):
        """Test single vectors if they parallel or antiprallel."""
        assert is_almost_parallel_or_antiprallel(np.array(v1), np.array(v2)) == result

    def test_is_almost_parallel_or_antiprallel_multiple_vector(self):
        """Test array of vectors."""
        v1 = np.repeat(np.array([1.0, 0, 0])[None, :], 4, axis=0)
        v2 = np.repeat(np.array([2.0, 0, 0])[None, :], 4, axis=0)
        goal = np.repeat((True), 4, axis=0)
        assert_array_equal(is_almost_parallel_or_antiprallel(v1, v2), goal)


class TestNormalize:
    """Test the function `normalize`."""

    def test_normalize_1d_array(self):
        """Test 1D array."""
        assert_array_equal(normalize(np.array([2, 0, 0])), np.array([1, 0, 0]))

    @pytest.mark.parametrize(
        "v1, v2", [([0, 2, 0], [0, 1, 0]), ([2, 0, 0], [1, 0, 0]), ([0.5, 0.5, 0], [0.707107, 0.707107, 0]),],
    )
    def test_normalize_2d_array(self, v1, v2):
        """Test 2D array."""
        assert_array_almost_equal(normalize(np.array(v1)), np.array(v2))

    def test_normalize_all_zeros(self):
        """Test vector [0, 0, 0]."""
        with pytest.raises(ValueError):
            normalize(np.array([0, 0, 0]))


class TestFindRandomOrthogonal:
    """Test the function `find_random_orthogonal`."""

    def test_find_random_orthogonal_general(self):
        """Test find orthogonal for general vector`."""
        v = np.array([0.5, 0.2, 1])
        orthogonal = find_random_orthogonal(v)
        assert_almost_equal(np.dot(orthogonal, v), 0)
        assert_almost_equal(norm(orthogonal), 1)

    @pytest.mark.parametrize(
        "vec", [[1, 0, 0], [2, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, -2, 0], [0, 0, 1], [0, 0, -1], [0, 0, -2]]
    )
    def test_find_random_orthogonal_special(self, vec):
        """Test find_random_orthogonal for  vectors parallel or antiparallel to [1,0,0],[0,1,0],[0,0,1]`."""
        v = np.array(vec)
        orthogonal = find_random_orthogonal(v)
        assert_almost_equal(np.dot(orthogonal, v), 0)
        assert_almost_equal(norm(orthogonal), 1)


class TestFindOrthogonal:
    """Test the function `find_orthogonal`."""

    @pytest.mark.parametrize(
        "v1, v2",
        [
            ([1, 0, 0], [-1, 0, 0]),
            ([1, 0, 0], [0, 1, 0]),
            ([2, 0, 0], [0, 1, 0]),
            ([1, 0, 0], [1, 0, 0]),
            ([1, 0.2, 1], [4, 1.2, 0]),
        ],
    )
    def test_find_orthogonal(self, v1, v2):
        """Test find_orthogonal for 1D vectors`."""
        v1 = np.array(v1)
        v2 = np.array(v2)
        orthogonal = find_orthogonal(v1, v2)
        assert_almost_equal(np.dot(orthogonal, v1), 0)
        assert_almost_equal(np.dot(orthogonal, v2), 0)
        assert_almost_equal(norm(orthogonal), 1)

    def test_find_orthogonal_array(self):
        """Test find_orthogonal for multidimension vectors`."""
        v1 = np.array(4 * [[1, 0, 0]])
        v2 = np.array(4 * [[0, 1, 0]])
        with pytest.raises(ValueError):
            find_orthogonal(v1, v2)


class TestFindUnsigned3dAngle:
    """Test the function `find_unsigned_3d_angle`."""

    @pytest.mark.parametrize(
        "v1, v2, result",
        [
            ([1, 0, 0], [0, 1, 0], np.pi / 2),
            ([2, 0, 0], [0, 2, 0], np.pi / 2),
            ([1, 0, 0], [1, 0, 0], 0),
            ([-1, 0, 0], [-1, 0, 0], 0),
            ([1, 0, 0], [-1, 0, 0], np.pi),
        ],
    )
    def test_find_unsigned_3d_angle(self, v1, v2, result):
        """Test  `find_unsigned_3d_angle` between two 1D vector."""
        v1 = np.array(v1)
        v2 = np.array(v2)
        assert_almost_equal(find_unsigned_3d_angle(v1, v2), result)

    def test_find_3d_angle_array(self):
        """Test  `find_unsigned_3d_angle` between two 2D vector."""
        v1 = np.array(4 * [[1, 0, 0]])
        v2 = np.array(4 * [[0, 1, 0]])
        output = find_unsigned_3d_angle(v1, v2)
        assert len(output) == 4
        assert_array_almost_equal(output, 4 * [np.pi / 2])
