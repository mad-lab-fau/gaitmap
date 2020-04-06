import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.spatial.transform import Rotation
from gaitmap.preprocessing.align_to_gravity import align


class TestAlignToGravity:
    """Test the function `align_to_gravity`."""

    def test_align_to_gravity(self):
        """Test some basic functionality."""
        test_data = np.array([1, 2, 3])
        align(test_data, Rotation.from_rotvec(np.pi / 2 * np.array([0, 0, 1])))
        assert True
