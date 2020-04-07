import numpy as np
import pandas as pd
import pytest
import scipy

from gaitmap.trajectory_reconstruction.position import ForwardBackwardIntegration
from gaitmap.utils.consts import SF_ACC, SF_COLS


class TestForwardBackwardIntegration:
    # TODO implement tests for turning_point != 0.5?
    def test_estimate_velocity_dummy_data(self):
        # algorithm parameters
        turning_point = 0.5
        steepness = 0.08

        # make point symmetrical fake data
        # -> should have zero as output
        # -> should give nearly the same result as complete forward or backward integration (if .5 as turning point)
        dummy_data = self._get_point_symmetrical_dummy_data()
        sampling_frequency_hz = 100

        position = ForwardBackwardIntegration(turning_point, steepness)
        position.estimate(dummy_data, sampling_frequency_hz)

        # is the result nearly equal to zero?
        np.testing.assert_array_almost_equal([0, 0, 0], position.velocity_.iloc[-1])

        # is the result nearly the same as forward integration?
        np.testing.assert_array_almost_equal([0, 0, 0, 0, 0, 0], scipy.integrate.cumtrapz(dummy_data, axis=0)[-1])

        # is the result nearly the same as backward integration?
        np.testing.assert_array_almost_equal([0, 0, 0, 0, 0, 0], scipy.integrate.cumtrapz(dummy_data[::-1], axis=0)[-1])

    def test_estimate_position_dummy_data(self):
        # algorithm parameters
        turning_point = 0.5
        steepness = 0.08

        data = self._get_point_symmetrical_dummy_data()
        sampling_frequency_hz = 100

        position = ForwardBackwardIntegration(turning_point, steepness)
        position.estimate(data, sampling_frequency_hz)
        np.testing.assert_array_almost_equal([0, 0, 0], position.position_.iloc[-1])

    def test_single_sensor_input(self, healthy_example_imu_data, healthy_example_stride_borders):
        """Dummy test to see if the algorithm is generally working on the example data"""
        # TODO add assert statement / regression test to check against previous result
        data_left = healthy_example_imu_data["left_sensor"]
        position = ForwardBackwardIntegration(0.5, 0.08)
        position.estimate(data_left, 204.8)
        return None

    def test_estimate_multiple_sensors_input(self, healthy_example_imu_data):
        # TODO: change as soon as multi-sensor is implemented
        """Test if error is raised correctly on invalid input data type"""
        data = healthy_example_imu_data
        position = ForwardBackwardIntegration(0.5, 0.08)
        with pytest.raises(NotImplementedError, match=r"Multisensor input is not supported yet"):
            position.estimate(data, 204.8)

    def test_estimate_valid_input_data(self):
        """Test if error is raised correctly on invalid input data type"""
        data = pd.DataFrame({"a": [0, 1, 2], "b": [3, 4, 5]})
        position = ForwardBackwardIntegration(0.5, 0.08)
        with pytest.raises(ValueError, match=r"Provided data set is not supported by gaitmap"):
            position.estimate(data, 204.8)

    def _get_point_symmetrical_dummy_data(self):
        dummy = np.linspace(0, 1, 1000)
        dummy_data = np.concatenate((dummy, -np.flip(dummy)))
        dummy_pd = pd.DataFrame(data=np.tile(dummy_data, 6).reshape(6, len(dummy) * 2).transpose(), columns=SF_COLS)
        return dummy_pd
