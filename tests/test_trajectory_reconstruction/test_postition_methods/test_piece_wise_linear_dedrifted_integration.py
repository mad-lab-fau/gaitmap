import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal

from gaitmap.base import BasePositionMethod, BaseType
from gaitmap.trajectory_reconstruction.position_methods import PieceWiseLinearDedriftedIntegration
from gaitmap.utils.array_handling import bool_array_to_start_end_array
from gaitmap.utils.consts import SF_ACC, SF_GYR
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin
from tests.test_trajectory_reconstruction.test_postition_methods.test_pos_method_mixin import (
    TestPositionMethodNoGravityMixin,
)


class MetaTestConfig:
    algorithm_class = PieceWiseLinearDedriftedIntegration

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data, healthy_example_stride_events) -> BaseType:
        position = PieceWiseLinearDedriftedIntegration()
        # Get enough samples from the signal to ensure a ZUPT
        position.estimate(healthy_example_imu_data["left_sensor"].iloc[:500], sampling_rate_hz=204.8)
        return position


class TestMetaFunctionality(MetaTestConfig, TestAlgorithmMixin):
    __test__ = True


class TestSimpleIntegrationsNoGravity(TestPositionMethodNoGravityMixin):
    __test__ = True

    def init_algo_class(self) -> BasePositionMethod:
        # For basic integration tests, we do not remove gravity
        return PieceWiseLinearDedriftedIntegration(gravity=None).set_params(zupt_detector__window_length_s=0.1)

    @pytest.mark.parametrize("acc", ([0, 0, 1], [1, 2, 3]))
    def test_symetric_velocity_integrations(self, acc):
        """All test data starts and ends at zero."""
        # we had to overwrite this test as the PieceWiseLinearDedriftedIntegration function requires some valid
        # zupt updates within the test data
        test = self.init_algo_class().set_params(zupt_detector__window_length_s=0.3, zupt_detector__metric="maximum")

        test_data = np.repeat(np.array(acc)[None, :], 10, axis=0)
        test_data = np.vstack((test_data, -test_data))
        test_data = pd.DataFrame(test_data, columns=SF_ACC)
        test_data[SF_GYR] = np.repeat(np.array([20, 20, 20])[None, :], len(test_data), axis=0)
        # add some zupt regions
        test_data.loc[0:2, SF_GYR + SF_ACC] = 0
        test_data.loc[test_data.index[-3:], SF_GYR + SF_ACC] = 0
        expected = np.zeros(3)
        test = test.estimate(test_data, 10)

        assert_array_equal(test.velocity_.to_numpy()[0], expected)
        assert_array_equal(test.velocity_.to_numpy()[-1], expected)

    @pytest.mark.parametrize("acc", ([0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 2, 0], [1, 2, 0], [1, 2, 3]))
    def test_all_axis(self, acc):
        """Test against the physics equation."""
        # we had to overwrite this test as the PieceWiseLinearDedriftedIntegration function requires some valid
        # zupt updates within the test data
        test = self.init_algo_class().set_params(zupt_detector__window_length_s=0.3, zupt_detector__metric="maximum")

        n_steps = 10
        n_zupt_samples = 3

        acc = np.array(acc)
        test_data = np.repeat(acc[None, :], n_steps, axis=0)
        # simulting forward backward walking
        test_data = np.vstack(
            (
                np.zeros((n_zupt_samples, 3)),
                test_data,
                -test_data,
                [0, 0, 0],
                -test_data,
                test_data,
                np.zeros((n_zupt_samples, 3)),
            )
        )
        test_data = pd.DataFrame(test_data, columns=SF_ACC)
        test_data[SF_GYR] = np.repeat(np.array([20, 20, 20])[None, :], len(test_data), axis=0)
        # add some zupt regions
        test_data.loc[0 : n_zupt_samples - 1, SF_GYR] = 0
        test_data.loc[test_data.index[-n_zupt_samples:], SF_GYR] = 0

        fs = 10
        test.estimate(test_data, fs)

        expected = np.zeros(3)
        assert_array_almost_equal(test.position_.to_numpy()[-1], expected)
        assert_array_almost_equal(test.velocity_.to_numpy()[-1], expected)

        # Test quarter point
        # The +0.5 and the 0.25 comes because of the trapezoid rule integration *the 0.25 because you integrate twice)
        expected_vel = (acc * (n_steps - 1) + 0.5 * acc) / fs
        expected_pos = (0.5 * acc * (n_steps - 1) ** 2 + 0.5 * acc * (n_steps - 1) + 0.25 * acc) / fs ** 2
        assert_array_almost_equal(test.velocity_.to_numpy()[n_steps + n_zupt_samples], expected_vel)
        assert_array_almost_equal(test.position_.to_numpy()[n_steps + n_zupt_samples], expected_pos)


class TestPieceWiseLinearDedriftedIntegration:
    """Test the position estimation class `PieceWiseLinearDedriftedIntegration`."""

    def test_drift_model_simple(self):
        """Run a simple example and estimate its drift model"""
        data = np.array(
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                3,
                4,
                4,
                4,
                4,
                5,
                5,
                5,
                5,
                5,
                6,
                6,
                6,
                6,
                6,
                6,
                6,
                6,
                6,
                7,
                7,
                7,
                7,
                9,
                9,
                9,
                9,
                9,
                9,
                9,
                9,
                9,
            ]
        )
        zupt = np.array(
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ]
        )
        expected_output = np.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.15384615,
                1.30769231,
                1.46153846,
                1.61538462,
                1.76923077,
                1.92307692,
                2.07692308,
                2.23076923,
                2.38461538,
                2.53846154,
                2.69230769,
                2.84615385,
                3.0,
                3.0,
                3.0,
                3.0,
                3.0,
                3.4,
                3.8,
                4.2,
                4.6,
                5.0,
                5.0,
                5.0,
                5.0,
                5.0,
                5.28571429,
                5.57142857,
                5.85714286,
                6.14285714,
                6.42857143,
                6.71428571,
                7.0,
                7.28571429,
                7.57142857,
                7.85714286,
                8.14285714,
                8.42857143,
                8.71428571,
                9.0,
                9.0,
                9.0,
                9.0,
                9.0,
                9.0,
                9.0,
                9.0,
                9.0,
            ]
        )

        estimated_drift_model = PieceWiseLinearDedriftedIntegration()._estimate_piece_wise_linear_drift_model(
            data, bool_array_to_start_end_array(zupt.astype(bool))
        )
        assert_almost_equal(estimated_drift_model, expected_output)

    def test_drift_model_multidimensional(self):
        data = np.column_stack([np.linspace(1, 10, 10), np.linspace(10, 20, 10), np.linspace(20, 10, 10)])
        zupt = np.array([[5, 10]])
        estimated_drift_model = PieceWiseLinearDedriftedIntegration()._estimate_piece_wise_linear_drift_model(
            data, zupt
        )

        assert_almost_equal(estimated_drift_model, data)

    def test_drift_model(self):
        """Test drift model on simple slope with different zupt edge conditions"""
        data = np.arange(20)
        zupt = np.repeat(False, 20)
        zupt[11:15] = True
        zupt[5:8] = True

        estimated_drift_model = PieceWiseLinearDedriftedIntegration()._estimate_piece_wise_linear_drift_model(
            data, bool_array_to_start_end_array(zupt.astype(bool))
        )
        assert_almost_equal(data, estimated_drift_model)

        estimated_drift_model = PieceWiseLinearDedriftedIntegration()._estimate_piece_wise_linear_drift_model(
            data, bool_array_to_start_end_array(~zupt.astype(bool))
        )
        assert_almost_equal(data, estimated_drift_model)

    def test_all_zupt_data(self):
        """Test drift model all zupt."""
        data = np.arange(20)
        zupt = np.repeat(True, 20)

        estimated_drift_model = PieceWiseLinearDedriftedIntegration()._estimate_piece_wise_linear_drift_model(
            data, bool_array_to_start_end_array(zupt.astype(bool))
        )
        assert_almost_equal(data, estimated_drift_model)

    def test_no_zupt_data(self):
        """Test drift model no zupts available."""
        data = np.arange(20)
        zupt = np.repeat(False, 20)
        with pytest.raises(ValueError, match=r".*No valid zupt regions available*"):
            PieceWiseLinearDedriftedIntegration()._estimate_piece_wise_linear_drift_model(
                data, bool_array_to_start_end_array(zupt.astype(bool))
            )
