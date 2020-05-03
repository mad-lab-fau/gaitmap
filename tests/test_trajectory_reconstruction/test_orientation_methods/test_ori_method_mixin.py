import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from gaitmap.base import BaseOrientationMethod
from gaitmap.utils.consts import SF_GYR, SF_COLS
from gaitmap.utils.dataset_helper import is_single_sensor_orientation_list


class TestOrientationMethodMixin:
    algorithm_class = None
    __test__ = False

    def init_algo_class(self) -> BaseOrientationMethod:
        raise NotImplementedError("Should be implemented by ChildClass")

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

        fs = 100

        sensor_data = np.repeat(np.array([0, 0, 0, *axis_to_rotate])[None, :], fs, axis=0) * np.rad2deg(np.pi)
        sensor_data = pd.DataFrame(sensor_data, columns=SF_COLS)
        gyr_integrator = self.init_algo_class()

        gyr_integrator.estimate(sensor_data, fs)
        rot_final = gyr_integrator.orientation_.iloc[-1]

        np.testing.assert_array_almost_equal(Rotation(rot_final).apply(vector_to_rotate), expected_result, decimal=1)
        assert len(gyr_integrator.orientation_) == fs + 1

    def test_idiot_update(self):
        test = self.init_algo_class()
        fs = 10
        sensor_data = np.repeat(np.array([0, 0, 0, 0, 0, 0])[None, :], fs, axis=0) * np.rad2deg(np.pi)
        sensor_data = pd.DataFrame(sensor_data, columns=SF_COLS)
        test.estimate(sensor_data, fs)
        np.testing.assert_array_equal(test.orientation_.iloc[-1], test.initial_orientation.as_quat())

    def test_output_formats(self):
        test = self.init_algo_class()
        fs = 10
        sensor_data = np.repeat(np.array([0, 0, 0, 0, 0, 0])[None, :], fs, axis=0) * np.rad2deg(np.pi)
        sensor_data = pd.DataFrame(sensor_data, columns=SF_COLS)
        test.estimate(sensor_data, fs)

        assert isinstance(test.orientation_object_, Rotation)
        assert len(test.orientation_object_) == len(sensor_data) + 1
        assert is_single_sensor_orientation_list(test.orientation_, s_id=False)
        assert len(test.orientation_) == len(sensor_data) + 1

    def test_single_stride_regression(self, healthy_example_imu_data, healthy_example_stride_events, snapshot):
        """Simple regression test with default parameters."""
        test = self.init_algo_class()
        fs = 204.8

        strides = healthy_example_stride_events["left_sensor"]
        start, end = int(strides.iloc[:1]["start"]), int(strides.iloc[:1]["end"])
        data = healthy_example_imu_data['left_sensor'].iloc[start:end]

        test.estimate(data, fs)

        snapshot.assert_match(test.orientation_, test.__class__.__name__)
