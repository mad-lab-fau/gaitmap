from numpy.testing import assert_almost_equal
import numpy as np
from gaitmap.utils.rotations import rotation_from_angle


class TestRotationFromAngle:
    def test_single_angle(self):
        """Test single axis, single angle."""
        assert_almost_equal(rotation_from_angle(np.array([1, 0, 0]), np.pi).as_quat(), [1.0, 0, 0, 0])

    def test_multiple_axis_and_angles(self):
        """Test multiple axis, multiple angles."""
        start = np.repeat(np.array([1.0, 0, 0])[None, :], 5, axis=0)
        goal = np.repeat(np.array([1.0, 0, 0, 0])[None, :], 5, axis=0)
        angle = np.array([np.pi] * 5)
        assert_almost_equal(rotation_from_angle(start, angle).as_quat(), goal)

    def test_multiple_axis_single_angle(self):
        """Test multiple axis, single angles."""
        start = np.repeat(np.array([1.0, 0, 0])[None, :], 5, axis=0)
        goal = np.repeat(np.array([1.0, 0, 0, 0])[None, :], 5, axis=0)
        angle = np.array(np.pi)
        assert_almost_equal(rotation_from_angle(start, angle).as_quat(), goal)

    def test_single_axis_multiple_angle(self):
        """Test single axis, multiple angles."""
        start = np.array([1.0, 0, 0])[None, :]
        goal = np.repeat(np.array([1.0, 0, 0, 0])[None, :], 5, axis=0)
        angle = np.array([np.pi] * 5)
        assert_almost_equal(rotation_from_angle(start, angle).as_quat(), goal)
