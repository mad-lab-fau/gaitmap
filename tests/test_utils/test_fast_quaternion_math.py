import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.spatial.transform.rotation import Rotation

from gaitmap.utils.fast_quaternion_math import (
    multiply,
    normalize,
    quat_from_rotvec,
    rate_of_change_from_gyro,
    rotate_vector,
)
from tests.test_utils.test_vector_math import TestNormalize


class TestRateOfChangeFromGyro:
    """Test the function `rate_of_change_from_gyro`."""

    @pytest.mark.parametrize(
        "q, g",
        [
            ([1.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0]),
            ([0.0, 0.707107, 0.707107, 0.0], [1.0, 2.0, 1.0]),
            ([0.5, 0.5, 0.5, 0.5], [1.0, 4.0, 0.4]),
        ],
    )
    def test_rate_of_change_from_gyro(self, q, g):
        q = np.array(q)
        g = np.array(g)
        assert_array_almost_equal(rate_of_change_from_gyro(g, q), 0.5 * multiply(q, np.append(g, 0.0)))


class TestMulitply:
    """Test the function `multiply`."""

    @pytest.mark.parametrize(
        "q1, q2",
        [
            ([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            ([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]),
            ([0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 0.707107, 0.707107]),
        ],
    )
    def test_quaternion_multiplication(self, q1, q2):
        q1 = np.array(q1)
        q2 = np.array(q2)
        assert_array_almost_equal(multiply(q1, q2), (Rotation.from_quat(q1) * Rotation.from_quat(q2)).as_quat())
        assert_array_almost_equal(multiply(q2, q1), (Rotation.from_quat(q2) * Rotation.from_quat(q1)).as_quat())


class TestRotateVector:
    """Test the function `rotate_vector`."""

    @pytest.mark.parametrize(
        "q, v",
        [
            ([1.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0]),
            ([0.0, 0.707107, 0.707107, 0.0], [1.0, 2.0, 1.0]),
            ([0.5, 0.5, 0.5, 0.5], [1.0, 4.0, 0.4]),
        ],
    )
    def test_rotate_vector_by_quaternion(self, q, v):
        q = np.array(q)
        v = np.array(v)
        assert_array_almost_equal(rotate_vector(q, v), Rotation.from_quat(q).apply(v))


class TestQuatFromRotvec:
    """Test the function `quat_from_rotvec`."""

    @pytest.mark.parametrize(
        "v", [([1.0, 0.0, 0.0]), ([1.0, 1.0, 0.0]), ([1.0, 1.0, 1.0]), ([0.2, 0.1, 5.0]), ([10.0, 0.2, 0.0])]
    )
    def test_quat_from_rotation_vector(self, v):
        """Test quat_from_rotation_vector`."""
        v = np.array(v)
        assert_array_almost_equal(quat_from_rotvec(v), Rotation.from_rotvec(v).as_quat())


class TestFindNormalize(TestNormalize):
    def func(self, x):
        return normalize(x)

    def test_normalize_all_zeros(self):
        """Test vector [0, 0, 0]."""
        assert_array_almost_equal(self.func(np.array([0.0, 0, 0])), [0.0, 0, 0])
