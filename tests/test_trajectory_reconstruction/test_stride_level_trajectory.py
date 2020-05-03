import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from scipy.spatial.transform import Rotation

from gaitmap.base import BaseType
from gaitmap.trajectory_reconstruction.stride_level_trajectory import StrideLevelTrajectory
from gaitmap.utils.consts import SF_COLS
from gaitmap.utils.rotations import find_angle_between_orientations
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin


class TestMetaFunctionality(TestAlgorithmMixin):
    algorithm_class = StrideLevelTrajectory
    __test__ = True

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data, healthy_example_stride_events) -> BaseType:
        trajectory = StrideLevelTrajectory()
        trajectory.estimate(
            healthy_example_imu_data["left_sensor"],
            healthy_example_stride_events["left_sensor"].iloc[:2],
            sampling_rate_hz=1,
        )
        return trajectory


class TestInitCalculation:
    """Test the calculation of initial rotations per stride.

    No complicated tests here, as this uses `get_gravity_rotation`, which is well tested
    """

    def test_calc_initial_dummy(self):
        """No rotation expected as already aligned."""
        dummy_data = pd.DataFrame(np.repeat(np.array([0, 0, 1, 0, 0, 0])[None, :], 20, axis=0), columns=SF_COLS)
        start_ori = StrideLevelTrajectory.calculate_initial_orientation(dummy_data, 10, 8)
        assert_array_equal(start_ori.as_quat(), Rotation.identity().as_quat())

    @pytest.mark.parametrize("start", [0, 99])
    def test_start_of_stride_equals_start_or_end_of_data(self, start):
        """If start is to close to the start or the end of the data a warning is emitted."""
        dummy_data = pd.DataFrame(np.repeat(np.array([0, 0, 1, 0, 0, 0])[None, :], 100, axis=0), columns=SF_COLS)
        with pytest.warns(UserWarning) as w:
            StrideLevelTrajectory.calculate_initial_orientation(dummy_data, start, 8)

        assert "complete window length" in str(w[0])
