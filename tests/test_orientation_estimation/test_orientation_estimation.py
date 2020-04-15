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

        fs = 100
        window_width = 8
        # start at window_with/2 because GyroIntegration._calculate_initial_orientation uses start+-half window size
        start_sample = int(np.floor(window_width / 2))
        # 180 degree rotation around i_axis
        sensor_data, event_list = self.get_dummy_data(start_sample, axis_to_rotate, fs)

        gyr_integrator = GyroIntegration(align_window_width=window_width)
        gyr_integrator.estimate(sensor_data, event_list, fs)
        rot_final = gyr_integrator.estimated_orientations_.iloc[-1]
        np.testing.assert_array_almost_equal(Rotation(rot_final).apply(vector_to_rotate), expected_result, decimal=1)
        assert len(gyr_integrator.estimated_orientations_) == fs + 1

    @staticmethod
    def get_dummy_data(start_sample, axis_to_rotate, fs):
        sensor_data = pd.DataFrame(columns=SF_COLS)
        for i_axis in SF_GYR:
            if i_axis == SF_GYR[axis_to_rotate]:
                sensor_data[i_axis] = [np.pi] * (fs + start_sample)
            else:
                sensor_data[i_axis] = [0] * (fs + start_sample)

        sensor_data[SF_ACC] = [0, 0, 1]
        event_list = pd.DataFrame(
            data=[[0, start_sample, start_sample + fs, 0, 0, start_sample, 0]],
            columns=["s_id", "start", "end", "pre_ic", "ic", "min_vel", "tc"],
        )
        return sensor_data, event_list

    def test_single_sensor_input(self, healthy_example_imu_data, healthy_example_stride_events):
        """Dummy test to see if the algorithm is generally working on the example data"""
        # TODO add assert statement / regression test to check against previous result
        gyr_int = self.estimate_one_sensor(healthy_example_imu_data, healthy_example_stride_events)
        gyr_int.estimated_orientations_without_final_
        gyr_int.estimated_orientations_without_initial_
        return None

    def test_orientations_without_initial(self, healthy_example_imu_data, healthy_example_stride_events):
        gyr_int = self.estimate_one_sensor(healthy_example_imu_data, healthy_example_stride_events)
        r = gyr_int.estimated_orientations_
        r_wo_initial = gyr_int.estimated_orientations_without_initial_

        deleted_idx_initial = r.index.difference(r_wo_initial.index)
        deleted_idx_sample_initial = deleted_idx_initial.get_level_values(level="sample")

        assert len(r_wo_initial), len(r) - 1
        assert len(deleted_idx_sample_initial.unique()) == 1
        assert deleted_idx_sample_initial.unique()[0] == 0
        for i_stride, i_stride_data in r_wo_initial.groupby(level="s_id"):
            i_stride_data.index.get_level_values(level="sample")[0] == 1

    def test_orientations_without_final(self, healthy_example_imu_data, healthy_example_stride_events):
        gyr_int = self.estimate_one_sensor(healthy_example_imu_data, healthy_example_stride_events)
        r = gyr_int.estimated_orientations_
        r_wo_final = gyr_int.estimated_orientations_without_final_

        assert len(r_wo_final), len(r) - 1
        for i_stride, i_stride_data in r_wo_final.groupby(level="s_id"):
            old_final = r.xs(i_stride, level="s_id",).index.get_level_values(level="sample")[-1]
            new_final = r_wo_final.index.get_level_values(level="sample")[-1]
            assert new_final, old_final - 1

    def test_multiple_sensor_input(self, healthy_example_imu_data, healthy_example_stride_events):
        """Dummy test to see if the algorithm is generally working on the example data"""
        # TODO add assert statement / regression test to check against previous result
        data = healthy_example_imu_data
        gyr_int = GyroIntegration(align_window_width=8)
        gyr_int.estimate(data, healthy_example_stride_events, 204.8)
        gyr_int.estimated_orientations_without_final_
        gyr_int.estimated_orientations_without_initial_
        return None

    def test_valid_input_data(self, healthy_example_imu_data, healthy_example_stride_events):
        """Test if error is raised correctly on invalid input data type"""
        data = healthy_example_imu_data
        stride_list = healthy_example_stride_events
        fake_data = pd.DataFrame({"a": [0, 1, 2], "b": [3, 4, 5]})
        fake_stride_list = {
            "a": pd.DataFrame(data=[[0, 1, 2]], columns=["stride", "begin", "stop"]),
            "b": pd.DataFrame(data=[[0, 1, 2]], columns=["stride", "begin", "stop"]),
        }

        gyr_int = GyroIntegration(align_window_width=8)

        with pytest.raises(ValueError, match=r"Provided data set is not supported by gaitmap"):
            gyr_int.estimate(fake_data, stride_list, 204.8)
        with pytest.raises(ValueError, match=r"Provided stride event list is not supported by gaitmap"):
            gyr_int.estimate(data, fake_stride_list, 204.8)

    def test_multi_sensor_output(self, healthy_example_imu_data, healthy_example_stride_events):
        data = healthy_example_imu_data
        gyr_int = GyroIntegration(align_window_width=8)
        gyr_int.estimate(data, healthy_example_stride_events, 204.8)
        results = [
            gyr_int.estimated_orientations_without_final_,
            gyr_int.estimated_orientations_without_initial_,
            gyr_int.estimated_orientations_,
        ]
        for i_result in results:
            assert isinstance(i_result, dict)
            assert i_result.keys, healthy_example_imu_data.keys

    def test_start_of_stride_equals_start_or_end_of_data(self):
        fs = 100
        stride_start = [0, fs - 1]
        window_width = 8
        for i_start in stride_start:
            sensor_data, event_list = self.get_dummy_data(start_sample=i_start, axis_to_rotate=1, fs=fs)
            gyr_integrator = GyroIntegration(align_window_width=window_width)
            gyr_integrator.estimate(sensor_data, event_list, fs)
            gyr_integrator.estimated_orientations_
            gyr_integrator.estimated_orientations_without_initial_
            gyr_integrator.estimated_orientations_without_final_
        return None

    def estimate_one_sensor(self, healthy_example_imu_data, healthy_example_stride_events) -> GyroIntegration:
        data_left = healthy_example_imu_data["left_sensor"]
        stride_events_left = healthy_example_stride_events["left_sensor"]
        gyr_int = GyroIntegration(align_window_width=8)
        gyr_int.estimate(data_left, stride_events_left, 204.8)
        return gyr_int
