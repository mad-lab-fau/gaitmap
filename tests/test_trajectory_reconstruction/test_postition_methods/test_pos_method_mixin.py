import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from pandas._testing import assert_frame_equal

from gaitmap.base import BasePositionMethod
from gaitmap.trajectory_reconstruction import SimpleGyroIntegration
from gaitmap.utils.consts import GF_POS, GF_VEL, SF_ACC, SF_COLS

# TODO: Add regression test on single stride
from gaitmap.utils.datatype_helper import is_single_sensor_position_list
from gaitmap.utils.rotations import rotate_dataset_series


class TestPositionMethodNoGravityMixin:
    __test__ = False

    def init_algo_class(self) -> BasePositionMethod:
        raise NotImplementedError("Should be implemented by ChildClass")

    def test_idiot_update(self):
        """Integrate zeros."""
        test = self.init_algo_class()
        idiot_data = pd.DataFrame(np.zeros((10, 6)), columns=SF_COLS)

        test = test.estimate(idiot_data, sampling_rate_hz=100)
        expected = np.zeros((11, 3))
        expected_vel = pd.DataFrame(expected, columns=GF_VEL)
        expected_vel.index.name = "sample"
        expected_pos = pd.DataFrame(expected, columns=GF_POS)
        expected_pos.index.name = "sample"

        assert_frame_equal(test.position_, expected_pos)
        assert_frame_equal(test.velocity_, expected_vel)
        assert_frame_equal(test.position_, expected_pos)

    def test_output_formats(self):
        test = self.init_algo_class()
        sensor_data = pd.DataFrame(np.zeros((10, 6)), columns=SF_COLS)

        test = test.estimate(sensor_data, sampling_rate_hz=100)

        assert is_single_sensor_position_list(test.position_, position_list_type=None)
        assert len(test.position_) == len(sensor_data) + 1
        assert len(test.velocity_) == len(sensor_data) + 1

    @pytest.mark.parametrize("acc", ([0, 0, 1], [1, 2, 3]))
    def test_symetric_velocity_integrations(self, acc):
        """All test data starts and ends at zero."""
        test = self.init_algo_class()

        test_data = np.repeat(np.array(acc)[None, :], 5, axis=0)
        test_data = np.vstack((test_data, -test_data))
        test_data = pd.DataFrame(test_data, columns=SF_ACC)

        test = test.estimate(test_data, sampling_rate_hz=1)
        expected = np.zeros(3)

        assert_array_equal(test.velocity_.to_numpy()[0], expected)
        assert_array_equal(test.velocity_.to_numpy()[-1], expected)

    @pytest.mark.parametrize("acc", ([0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 2, 0], [1, 2, 0], [1, 2, 3]))
    def test_all_axis(self, acc):
        """Test against the physics equation."""
        test = self.init_algo_class()

        n_steps = 10

        acc = np.array(acc)
        test_data = np.repeat(acc[None, :], n_steps, axis=0)
        test_data = np.vstack((test_data, -test_data, [0, 0, 0]))
        # This simulates forward and backwards walking
        test_data = np.vstack((test_data, -test_data))
        test_data = pd.DataFrame(test_data, columns=SF_ACC)

        test.estimate(test_data, sampling_rate_hz=1)

        expected = np.zeros(3)
        assert_array_almost_equal(test.position_.to_numpy()[-1], expected)
        assert_array_almost_equal(test.velocity_.to_numpy()[-1], expected)

        # Test quarter point
        # The +0.5 and the 0.25 comes because of the trapezoid rule integration *the 0.25 because you integrate twice)
        expected_vel = acc * (n_steps - 1) + 0.5 * acc
        expected_pos = 0.5 * acc * (n_steps - 1) ** 2 + 0.5 * acc * (n_steps - 1) + 0.25 * acc
        assert_array_almost_equal(test.velocity_.to_numpy()[n_steps], expected_vel)
        assert_array_almost_equal(test.position_.to_numpy()[n_steps], expected_pos)

    def test_single_stride_regression(self, healthy_example_imu_data, healthy_example_stride_events, snapshot):
        """Simple regression test with default parameters."""
        test = self.init_algo_class()
        fs = 204.8

        strides = healthy_example_stride_events["left_sensor"]
        start, end = int(strides.iloc[:1]["start"]), int(strides.iloc[:1]["end"])
        data = healthy_example_imu_data["left_sensor"].iloc[start:end]
        orientation = SimpleGyroIntegration().estimate(data, sampling_rate_hz=fs).orientation_object_
        data = rotate_dataset_series(data, orientation[:-1])

        test.estimate(data, sampling_rate_hz=fs)

        snapshot.assert_match(test.position_, test.__class__.__name__)
