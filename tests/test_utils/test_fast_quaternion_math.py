import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.spatial.transform import Rotation

from gaitmap.utils.fast_quaternion_math import q_multiply
from gaitmap.utils.rotations import find_angle_between_orientations


class TestQuatMultiply:
    @pytest.mark.parametrize("p, q", (([0, 0, 0, 1.0], [0, 0, 0, 1.0]), ([1.0, 0, 0, 0], [0, 1.0, 0, 0.0])))
    def test_against_rotation(self, p, q):
        """Test directly against to multiply function of scipy rotations.
        """
        p, q = np.array(p), np.array(q)
        out = q_multiply(p, q)
        expected = Rotation.from_quat(p) * Rotation.from_quat(q)

        assert find_angle_between_orientations(Rotation.from_quat(out), expected) == 0.0

    # TODO: 2D test
