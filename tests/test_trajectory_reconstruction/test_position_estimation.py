import numpy as np
import pandas as pd
import pytest
import scipy

from gaitmap.base import BaseType
from gaitmap.trajectory_reconstruction.orientation_estimation import GyroIntegration
from gaitmap.trajectory_reconstruction.position_estimation import ForwardBackwardIntegration
from gaitmap.utils.consts import SF_COLS, SF_ACC, SF_VEL, SF_POS
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin


class TestMetaFunctionality(TestAlgorithmMixin):
    algorithm_class = ForwardBackwardIntegration
    __test__ = True

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data, healthy_example_stride_events) -> BaseType:
        position = ForwardBackwardIntegration()
        position.estimate(
            healthy_example_imu_data["left_sensor"],
            healthy_example_stride_events["left_sensor"].iloc[:2],
            sampling_rate_hz=1,
        )
        return position


class TestForwardBackwardIntegration:
    # TODO implement tests for turning_point != 0.5?
    start_sample = 5
    dummy_data_length = 1000

    def test_estimate_velocity_dummy_data(self):
        # algorithm parameters
        turning_point = 0.5
        steepness = 0.08
        # sampling rate is arbitrary / should not influence this test
        fs = 100

        # make point symmetrical fake data
        # -> should have zero as output, when integrating it using single DRI
        # -> should give nearly the same result as complete forward or backward integration (if .5 as turning point)
        dummy_data = self._get_dummy_data(self.dummy_data_length, "point-symmetrical")

        # get dummy event list for one stride
        event_list = self._get_dummy_event_list(dummy_data)

        # don't rotate data
        rots = pd.DataFrame([[0, 0, 0, 0, 1]]*(len(dummy_data)-self.start_sample), columns=["s_id", "qx", "qy", "qz",
                                                                                         "qw"])
        rots.set_index("s_id", append=True, inplace=True)
        position = ForwardBackwardIntegration(turning_point, steepness)
        position.estimate(dummy_data, event_list, rots, fs)

        # is the result nearly equal to zero?
        np.testing.assert_array_almost_equal([0, 0, 0], position.estimated_velocity_.iloc[-1])

        # Deleted comparison to simple forward OR backward integration since it is not almost equal to the
        # forward-backward integration (simple forward integration results in [0.01, 0.01, 0.01]

    def test_estimate_position_dummy_data(self):
        # algorithm parameters
        turning_point = 0.5
        steepness = 0.08

        data = self._get_dummy_data(self.dummy_data_length, "non-symmetrical")
        events = self._get_dummy_event_list(data)
        np.linspace(0, 1, len(data))
        sampling_frequency_hz = 100

        position = ForwardBackwardIntegration(turning_point, steepness)
        position.estimate(data, events, sampling_frequency_hz)
        # TODO: use different test data, where just vertical will be zero
        final_position = position.estimated_position_.iloc[-1]
        np.testing.assert_almost_equal(final_position[2], 0)

    def test_single_sensor_input(self, healthy_example_imu_data, healthy_example_stride_events, snapshot):
        """Dummy test to see if the algorithm is generally working on the example data"""
        # TODO add assert statement / regression test to check against previous result
        data_left = healthy_example_imu_data["left_sensor"]
        events_left = healthy_example_stride_events["left_sensor"]
        position = ForwardBackwardIntegration()
        # TODO: Maybe the next two lines should be fixtures to not run into problmes when GyroIntegration is not
        #  working?
        rots = self.get_rotations(data_left, events_left, 204.8)
        position.estimate(data_left, events_left, rots, 204.8)
        snapshot.assert_match(position.estimated_position_.loc[:5], "pos")
        snapshot.assert_match(position.estimated_velocity_.loc[:5], "vel")

    def test_single_sensor_output(self, healthy_example_imu_data, healthy_example_stride_events):
        """Test if the output format is as expected for a single sensor"""
        data_left = healthy_example_imu_data["left_sensor"]
        events_left = healthy_example_stride_events["left_sensor"]
        rots = self.get_rotations(data_left, events_left, 204.8)
        position = ForwardBackwardIntegration()
        position.estimate(data_left, events_left, rots, 204.8)
        vel = position.estimated_velocity_
        pos = position.estimated_position_
        assert isinstance(vel, pd.DataFrame)
        assert isinstance(pos, pd.DataFrame)
        pd.testing.assert_index_equal(vel.columns, pd.Index(SF_VEL))
        pd.testing.assert_index_equal(pos.columns, pd.Index(SF_POS))

    def test_estimate_multi_sensors_input(self, healthy_example_imu_data, healthy_example_stride_events, snapshot):
        data = healthy_example_imu_data
        stride_events = healthy_example_stride_events
        rots = self.get_rotations(data, stride_events, 204.8)
        position = ForwardBackwardIntegration()
        position.estimate(data, stride_events, rots, 204.8)
        # Only comparing the first stride of pos, to keep the snapshot size manageable
        first_left = stride_events["left_sensor"].iloc[0]["s_id"]
        first_right = stride_events["right_sensor"].iloc[0]["s_id"]
        snapshot.assert_match(position.estimated_position_["left_sensor"].loc[first_left], "left")
        snapshot.assert_match(position.estimated_position_["right_sensor"].loc[first_right], "right")

    def test_multi_sensor_output(self, healthy_example_imu_data, healthy_example_stride_events):
        """Test if the output format is as expected for multi sensor"""
        data = healthy_example_imu_data
        stride_events = healthy_example_stride_events
        rots = self.get_rotations(data, stride_events, 204.8)
        position = ForwardBackwardIntegration()
        position.estimate(data, stride_events, rots, 204.8)
        vel = position.estimated_velocity_
        pos = position.estimated_position_
        assert isinstance(vel, dict)
        assert isinstance(pos, dict)
        assert vel.keys, healthy_example_imu_data.keys
        assert pos.keys, healthy_example_imu_data.keys

    def test_estimate_valid_input_data(self):
        """Test if error is raised correctly on invalid input data type"""
        data = pd.DataFrame({"a": [0, 1, 2], "b": [3, 4, 5]})
        events = []
        position = ForwardBackwardIntegration(0.6, 0.08)
        with pytest.raises(ValueError, match=r"Provided data set is not supported by gaitmap"):
            position.estimate(data, events, [], 204.8)

    @pytest.mark.parametrize("turning_point", (-0.1, 1.1))
    def test_bad_turning_point(self, turning_point: float, healthy_example_imu_data, healthy_example_stride_events):
        """Test if error is raised correctly on invalid input variable range"""
        data = healthy_example_imu_data
        stride_events = healthy_example_stride_events
        rots = self.get_rotations(data, stride_events, 204.8)
        with pytest.raises(ValueError, match=r"Turning point must be in the rage of 0.0 to 1.0"):
            position = ForwardBackwardIntegration(turning_point, 0.08)
            position.estimate(data, stride_events, rots, 204.8)

    def _get_dummy_data(self, length, style: str):
        dummy = np.linspace(0, 1, length)
        if style == "point-symmetrical":
            dummy_data = np.concatenate((dummy, -np.flip(dummy)))
            dummy_pd = pd.DataFrame(data=np.tile(dummy_data, 6).reshape(6, len(dummy) * 2).transpose(), columns=SF_COLS)
        else:
            if style == "non-symmetrical":
                dummy_data = dummy
                dummy_pd = pd.DataFrame(data=np.tile(dummy_data, 6).reshape(6, len(dummy)).transpose(), columns=SF_COLS)
            else:
                dummy_pd = pd.DataFrame()
        # we are faking a stride that begins at sample `start_sample` and thus we need the beginning of the
        #   point-symmetrical signal to that sample
        first_rows = dummy_pd.iloc[0 : self.start_sample]
        dummy_data = first_rows.append(dummy_pd, ignore_index=True)
        return dummy_pd

    def _get_dummy_event_list(self, dummy_data):
        return pd.DataFrame(data=[[0, self.start_sample, len(dummy_data)]], columns=["s_id", "start", "end"])

    def test_regression_position(self, healthy_example_imu_data, healthy_example_stride_events):
        """Just to see intermediate results so we know if we are on the right way"""
        data = healthy_example_imu_data
        stride_events = healthy_example_stride_events
        rots = self.get_rotations(data, stride_events, 204.8)
        position = ForwardBackwardIntegration()
        position.estimate(data, stride_events, rots, 204.8)
        pos = position.estimated_position_
        sls = []
        for i_stride in pos["left_sensor"].index.get_level_values(level="s_id").unique():
            sl = pos["left_sensor"].xs(int(i_stride), level="s_id").iloc[-1]
            sls.append(np.sqrt(np.square(sl).sum()))
        return

    @staticmethod
    def get_rotations(data, events, fs):
        gyr_integrator = GyroIntegration(align_window_width=8)
        gyr_integrator.estimate(data, events, fs)
        return gyr_integrator.estimated_orientations_without_final_

    # TODO: Add test for .rotate_stride
