import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from pandas._testing import assert_frame_equal
from scipy.spatial.transform import Rotation

from gaitmap.base import BaseTrajectoryMethod
from gaitmap.utils.consts import GF_ORI, GF_POS, GF_VEL, SF_COLS
from gaitmap.utils.datatype_helper import (
    is_single_sensor_orientation_list,
    is_single_sensor_position_list,
    is_single_sensor_velocity_list,
)
from gaitmap.utils.rotations import find_shortest_rotation, rotate_dataset


class TestTrajectoryMethodMixin:
    __test__ = False

    def init_algo_class(self) -> BaseTrajectoryMethod:
        raise NotImplementedError("Should be implemented by ChildClass")

    def test_idiot_update(self):
        """Integrate zeros except for gravity."""
        test = self.init_algo_class()
        idiot_data = pd.DataFrame(np.zeros((15, 6)), columns=SF_COLS)
        idiot_data["acc_z"] = 9.81
        test = test.estimate(idiot_data, 15)
        expected = np.zeros((16, 3))
        expected_vel = pd.DataFrame(expected, columns=GF_VEL)
        expected_vel.index.name = "sample"
        expected_pos = pd.DataFrame(expected, columns=GF_POS)
        expected_pos.index.name = "sample"
        orientations = np.repeat(test.initial_orientation[None, :], 16, axis=0)
        expected_ori = pd.DataFrame(orientations, columns=GF_ORI)
        expected_ori["q_w"] = 1.0
        expected_ori.index.name = "sample"

        assert_frame_equal(test.position_, expected_pos)
        assert_frame_equal(test.velocity_, expected_vel)
        assert_frame_equal(test.orientation_, expected_ori)

    def test_output_formats(self):
        test = self.init_algo_class()
        fs = 15
        sensor_data = np.repeat(np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])[None, :], fs, axis=0)
        sensor_data = pd.DataFrame(sensor_data, columns=SF_COLS)
        test.estimate(sensor_data, fs)

        assert isinstance(test.orientation_object_, Rotation)
        assert len(test.orientation_object_) == len(sensor_data) + 1
        assert is_single_sensor_orientation_list(test.orientation_, orientation_list_type=None)
        assert len(test.orientation_) == len(sensor_data) + 1
        assert is_single_sensor_position_list(test.position_, position_list_type=None)
        assert len(test.position_) == len(sensor_data) + 1
        assert is_single_sensor_velocity_list(test.velocity_, velocity_list_type=None)
        assert len(test.velocity_) == len(sensor_data) + 1

    @pytest.mark.parametrize(
        "axis_to_rotate, vector_to_rotate, expected_result",
        (([1, 0, 0], [0, 0, 1], [0, 0, -1]), ([0, 1, 0], [0, 0, 1], [0, 0, -1]), ([0, 0, 1], [1, 0, 0], [-1, 0, 0])),
    )
    def test_180(self, axis_to_rotate: int, vector_to_rotate: list, expected_result: list):
        """Rotate by 180 degree around one axis and check resulting rotation by transforming a 3D vector with start
        and final rotation.

        Parameters
        ----------
        axis_to_rotate
            the axis around which should be rotated

        vector_to_rotate
            test vector that will be transformed using the initial orientation and the updated/final orientation

        expected_result
            the result that is to be expected

        """

        fs = 100.0

        sensor_data = np.repeat(np.array([0.0, 0.0, 0.0, *axis_to_rotate])[None, :], fs, axis=0) * np.rad2deg(np.pi)
        sensor_data = pd.DataFrame(sensor_data, columns=SF_COLS)
        test = self.init_algo_class()

        test.estimate(sensor_data, fs)
        rot_final = test.orientation_.iloc[-1]

        np.testing.assert_array_almost_equal(Rotation(rot_final).apply(vector_to_rotate), expected_result, decimal=1)
        assert len(test.orientation_) == fs + 1

    def test_symmetric_velocity_integrations(self):
        """Test data starts and ends at zero."""
        test = self.init_algo_class()
        acc = np.array([0.0, 0.0, 10.0])
        g = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 40.0])  # to prevent zupts
        accel_data = np.repeat(np.concatenate((g + acc, gyro))[None, :], 15, axis=0)
        break_data = np.repeat(np.concatenate((g - acc, gyro))[None, :], 15, axis=0)
        test_data = np.vstack((accel_data, break_data))
        test_data = pd.DataFrame(test_data, columns=SF_COLS)

        test = test.estimate(test_data, 10)
        expected = np.zeros(3)

        assert_array_almost_equal(test.velocity_.to_numpy()[0], expected, decimal=10)
        assert_array_almost_equal(test.velocity_.to_numpy()[-1], expected, decimal=10)

    def test_full_trajectory_regression(self, healthy_example_imu_data, snapshot):
        """Simple regression test with default parameters."""
        test = self.init_algo_class()
        fs = 204.8
        data = healthy_example_imu_data["left_sensor"].iloc[:3000]

        initial_g = np.median(data.to_numpy()[:100, :3], axis=0)
        initial_rotation = find_shortest_rotation(initial_g / np.linalg.norm(initial_g), np.array([0, 0, 1]))
        data = rotate_dataset(data, initial_rotation)

        test.estimate(data, fs)

        snapshot.assert_match(test.position_, test.__class__.__name__)
