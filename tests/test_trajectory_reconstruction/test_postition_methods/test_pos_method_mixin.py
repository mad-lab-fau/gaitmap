import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas._testing import assert_frame_equal

from gaitmap.base import BasePositionMethod
from gaitmap.utils.consts import SF_COLS, SF_ACC, GF_VEL, GF_POS


class TestPositionMethodNoGravityMixin:
    # TODO: Add simple single stride regression test
    algorithm_class = None
    __test__ = False

    def init_algo_class(self) -> BasePositionMethod:
        raise NotImplementedError("Should be implemented by ChildClass")

    def test_idiot_update(self):
        """Integrate zeros."""
        test = self.init_algo_class()
        idiot_data = pd.DataFrame(np.zeros((10, 6)), columns=SF_COLS)

        test = test.estimate(idiot_data, 1)
        expected = np.zeros((10, 3))
        expected_vel = pd.DataFrame(expected, columns=GF_VEL)
        expected_pos = pd.DataFrame(expected, columns=GF_POS)

        assert_frame_equal(test.position_, expected_pos)
        assert_frame_equal(test.velocity_, expected_vel)
        assert_frame_equal(test.position_list_, expected_pos)

    @pytest.mark.parametrize("acc", ([0, 0, 1], [1, 2, 3]))
    def test_symetric_velocity_integrations(self, acc):
        """All test data starts and ends at zero."""
        test = self.init_algo_class()

        test_data = np.repeat(np.array(acc)[None, :], 5, axis=0)
        test_data = np.vstack((test_data, -test_data))
        test_data = pd.DataFrame(test_data, columns=SF_ACC)

        test = test.estimate(test_data, 1)
        expected = np.zeros(3)

        assert_array_equal(test.velocity_.to_numpy()[0], expected)
        assert_array_equal(test.velocity_.to_numpy()[-1], expected)

    @pytest.mark.parametrize("acc", (
            [0, 1, 0],
            [1, 0, 0],
            [0, 2, 0],
            [1, 2, 0],
    ))
    def test_forward_all_axis_with_level_walk(self, acc):
        """Test against the physics equation."""
        test = self.init_algo_class()

        n_steps = 2

        acc = np.array(acc)
        test_data = np.repeat(acc[None, :], n_steps, axis=0)
        test_data = np.vstack((test_data, -test_data))
        test_data = pd.DataFrame(test_data, columns=SF_ACC)

        test.estimate(test_data, 1)

        expected = acc * (n_steps - 1) * n_steps

        assert_array_equal(test.position_.to_numpy()[-1], expected)

