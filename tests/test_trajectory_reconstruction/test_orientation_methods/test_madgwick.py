import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from gaitmap.base import BaseOrientationMethod, BaseType
from gaitmap.trajectory_reconstruction import MadgwickAHRS, MadgwickRtsKalman
from gaitmap.trajectory_reconstruction.orientation_methods._madgwick import _madgwick_update, _madgwick_update_mag
from gaitmap.utils.consts import SF_COLS
from gaitmap.utils.exceptions import ValidationError
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin
from tests.mixins.test_caching_mixin import TestCachingMixin
from tests.test_trajectory_reconstruction.test_orientation_methods.test_ori_method_mixin import (
    TestOrientationMethodMixin,
)


class MetaTestConfig:
    algorithm_class = MadgwickAHRS

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data, healthy_example_stride_events) -> BaseType:
        position = MadgwickAHRS()
        position.estimate(healthy_example_imu_data["left_sensor"].iloc[:10], sampling_rate_hz=1)
        return position


class TestMetaFunctionality(MetaTestConfig, TestAlgorithmMixin):
    __test__ = True


class TestCachingFunctionality(MetaTestConfig, TestCachingMixin):
    __test__ = True


class TestSimpleRotations(TestOrientationMethodMixin):
    __test__ = True

    def init_algo_class(self) -> BaseOrientationMethod:
        return MadgwickAHRS()

    def test_correction_works(self) -> None:
        """Madgwick should be able to resist small roations if acc does not change."""
        ori = np.array([0, 0, 0, 1.0])
        initial_ori = ori
        for _i in range(50):
            ori = _madgwick_update(
                np.array([1, 1, 0.0]), np.array([0, 0.0, 1.0]), initial_orientation=ori, sampling_rate_hz=50, beta=1.0
            )

        np.testing.assert_array_almost_equal(ori, initial_ori, decimal=2)


class TestSimpleRotationsWithMag(TestOrientationMethodMixin):
    __test__ = True

    def init_algo_class(self) -> BaseOrientationMethod:
        return MadgwickAHRS(use_magnetometer=True)

    def test_raises_if_no_mag(self):
        with pytest.raises(ValidationError, match=".*'mag_x', 'mag_y', 'mag_z'.*"):
            test_data = pd.DataFrame(columns=SF_COLS)
            test = self.init_algo_class()
            test.estimate(test_data, sampling_rate_hz=10)

    def test_mag_does_something(self):
        """Madgwick should be able to resist small roations if acc does not change."""
        ori = np.array([0, 0, 0, 1.0])
        # With max in y
        for _i in range(50):
            ori_with_y = _madgwick_update_mag(
                np.array([1, 1, 0.0]),
                np.array([0, 0.0, 1.0]),
                np.array([0, 1.0, 0.0]),
                initial_orientation=ori,
                sampling_rate_hz=50,
                beta=1.0,
            )

        for _i in range(50):
            ori_with_x = _madgwick_update_mag(
                np.array([1, 1, 0.0]),
                np.array([0, 0.0, 1.0]),
                np.array([1.0, 0.0, 0.0]),
                initial_orientation=ori,
                sampling_rate_hz=50,
                beta=1.0,
            )

        with pytest.raises(AssertionError):
            # This is the stupid way to test "is the orientation different"
            np.testing.assert_array_almost_equal(ori_with_x, ori_with_y, 2)

    def test_single_stride_regression(self):
        # Overwrite, as the regression data has no mag data
        pass


@pytest.mark.parametrize("algorithm", [MadgwickAHRS, MadgwickRtsKalman])
def test_zero_magnetometer_retains_imu_estimate(algorithm):
    data = pd.DataFrame(0.0, index=range(30), columns=SF_COLS)
    data["acc_z"] = 9.81
    data[["gyr_x", "gyr_y", "gyr_z"]] = [3.0, -2.0, 1.0]
    data = data.assign(mag_x=0.0, mag_y=0.0, mag_z=0.0)
    initial = Rotation.from_euler("x", 20, degrees=True).as_quat()
    expected = algorithm(initial_orientation=initial).estimate(data, sampling_rate_hz=100)
    actual = algorithm(initial_orientation=initial, use_magnetometer=True).estimate(data, sampling_rate_hz=100)
    np.testing.assert_allclose(actual.orientation_, expected.orientation_, atol=1e-12)
    if algorithm is MadgwickRtsKalman:
        np.testing.assert_allclose(actual.position_, expected.position_, atol=1e-12)
        np.testing.assert_allclose(actual.velocity_, expected.velocity_, atol=1e-12)


def test_magnetometer_dropout_preserves_heading_and_resumes_correction():
    data = pd.DataFrame(0.0, index=range(100), columns=SF_COLS)
    data["acc_z"] = 9.81
    data = data.assign(mag_x=1.0, mag_y=0.0, mag_z=0.0)
    data.loc[30:59, "mag_x"] = 0.0
    initial = Rotation.from_euler("z", 90, degrees=True)
    result = MadgwickAHRS(initial_orientation=initial, use_magnetometer=True).estimate(data, sampling_rate_hz=100)
    yaw = result.orientation_object_.as_euler("xyz", degrees=True)[:, 2]
    np.testing.assert_allclose(yaw[31:61], yaw[30], atol=1e-10)
    assert yaw[-1] < yaw[30] < 90
