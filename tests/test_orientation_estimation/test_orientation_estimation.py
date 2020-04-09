import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from gaitmap.utils.consts import SF_COLS, SF_GYR, SF_ACC
from gaitmap.trajectory_reconstruction.orientation_estimation import GyroIntegration

# TODO: @to08kece @Arne, add metatest once DTW is merged


class TestGyroIntegration:
    @pytest.mark.parametrize(
        "axis_to_rotate, vector_to_rotate, expected_result",
        ((0, [0, 0, 1], [0, 0, -1]), (1, [0, 0, 1], [0, 0, -1]), (2, [1, 0, 0], [-1, 0, 0])),
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
        sensor_data = pd.DataFrame(columns=SF_COLS)
        fs = 100
        # start at 5 because GyroIntegration._calculate_initial_orientation uses start-4:start+4 samples
        start_sample = 5
        # 180 degree rotation around i_axis
        for i_axis in SF_GYR:
            if i_axis == SF_GYR[axis_to_rotate]:
                sensor_data[i_axis] = [np.pi] * (fs + start_sample)
            else:
                sensor_data[i_axis] = [0] * (fs + start_sample)

        sensor_data[SF_ACC] = [0, 0, 1]

        gyr_integrator = GyroIntegration()
        event_list = pd.DataFrame(data=[[0, start_sample, start_sample + fs]], columns=["s_id", "start", "end"])
        gyr_integrator.estimate(sensor_data, event_list, fs)
        rot_final = gyr_integrator.estimated_orientations_with_initial_.iloc[-1]
        np.testing.assert_array_almost_equal(Rotation(rot_final).apply(vector_to_rotate), expected_result, decimal=1)
        assert len(gyr_integrator.estimated_orientations_with_initial_) == fs + 1
      #  assert len(gyr_integrator.estimated_orientations_) == fs

    def test_single_sensor_input(self, healthy_example_imu_data, healthy_example_stride_events):
        """Dummy test to see if the algorithm is generally working on the example data"""
        # TODO add assert statement / regression test to check against previous result
        data_left = healthy_example_imu_data["left_sensor"]
        stride_events_left = healthy_example_stride_events["left_sensor"]
        gyr_int = GyroIntegration()
        gyr_int.estimate(data_left, stride_events_left, 204.8)

        return None

    def test_multiple_sensor_input(self, healthy_example_imu_data, healthy_example_stride_events):
        """Dummy test to see if the algorithm is generally working on the example data"""
        # TODO add assert statement / regression test to check against previous result
        data = healthy_example_imu_data
        gyr_int = GyroIntegration()
        gyr_int.estimate(data, healthy_example_stride_events, 204.8)

        return None

    def test_valid_input_data(self, healthy_example_imu_data):
        """Test if error is raised correctly on invalid input data type"""
        data = pd.DataFrame({"a": [0, 1, 2], "b": [3, 4, 5]})
        gyr_int = GyroIntegration()
        with pytest.raises(ValueError, match=r"Provided data set is not supported by gaitmap"):
            gyr_int.estimate(data, healthy_example_imu_data, 204.8)
