import numpy as np
import pandas as pd
import pytest

from gaitmap.base import BaseOrientationMethod, BaseType
from gaitmap.example_data import get_magnetometer_l_path_data
from gaitmap.trajectory_reconstruction import MahonyAHRS
from gaitmap.trajectory_reconstruction.orientation_methods._mahony import _mahony_update, _mahony_update_mag
from gaitmap.utils.consts import SF_COLS
from gaitmap.utils.exceptions import ValidationError
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin
from tests.mixins.test_caching_mixin import TestCachingMixin
from tests.test_trajectory_reconstruction.test_orientation_methods.test_ori_method_mixin import (
    TestOrientationMethodMixin,
)


def _rotation_angle_to_identity(quat):
    quat = np.asarray(quat, dtype=float)
    return 2 * np.arccos(np.clip(np.abs(quat[3]), -1.0, 1.0))


def _rotation_angle_between(quat_a, quat_b):
    quat_a = np.asarray(quat_a, dtype=float)
    quat_b = np.asarray(quat_b, dtype=float)
    dot = np.clip(np.abs(np.dot(quat_a, quat_b)), -1.0, 1.0)
    return 2 * np.arccos(dot)


class MetaTestConfig:
    algorithm_class = MahonyAHRS

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data, healthy_example_stride_events) -> BaseType:
        position = MahonyAHRS()
        position.estimate(healthy_example_imu_data["left_sensor"].iloc[:10], sampling_rate_hz=1)
        return position


class TestMetaFunctionality(MetaTestConfig, TestAlgorithmMixin):
    __test__ = True


class TestCachingFunctionality(MetaTestConfig, TestCachingMixin):
    __test__ = True


class TestSimpleRotations(TestOrientationMethodMixin):
    __test__ = True

    def init_algo_class(self) -> BaseOrientationMethod:
        return MahonyAHRS()

    def test_correction_works(self) -> None:
        """Mahony should be able to resist small rotations if acc does not change."""
        ori_without_correction = np.array([0, 0, 0, 1.0])
        ori_with_correction = np.array([0, 0, 0, 1.0])
        integral_error = np.zeros(3)
        for _i in range(50):
            ori_with_correction, integral_error = _mahony_update(
                np.array([1, 1, 0.0]),
                np.array([0, 0.0, 1.0]),
                initial_orientation=ori_with_correction,
                sampling_rate_hz=50,
                kp=10.0,
                ki=0.0,
                integral_error=integral_error,
            )
            ori_without_correction, _ = _mahony_update(
                np.array([1, 1, 0.0]),
                np.array([0, 0.0, 1.0]),
                initial_orientation=ori_without_correction,
                sampling_rate_hz=50,
                kp=0.0,
                ki=0.0,
                integral_error=np.zeros(3),
            )

        assert _rotation_angle_to_identity(ori_with_correction) < _rotation_angle_to_identity(ori_without_correction)

    def test_integral_term_is_sampling_rate_invariant(self) -> None:
        """The integral correction should model continuous-time integration and not depend on the sampling rate."""
        total_time_s = 4.0
        orientations = []
        for sampling_rate_hz in [50, 100, 200]:
            orientation = np.array([0, 0, 0, 1.0])
            integral_error = np.zeros(3)
            for _i in range(int(total_time_s * sampling_rate_hz)):
                orientation, integral_error = _mahony_update(
                    np.array([0.1, 0.0, 0.0]),
                    np.array([0, 0.0, 1.0]),
                    initial_orientation=orientation,
                    sampling_rate_hz=sampling_rate_hz,
                    kp=0.0,
                    ki=0.1,
                    integral_error=integral_error,
                )
            orientations.append(orientation)

        assert _rotation_angle_between(orientations[0], orientations[1]) < 1e-3
        assert _rotation_angle_between(orientations[1], orientations[2]) < 1e-3


class TestSimpleRotationsWithMag(TestOrientationMethodMixin):
    __test__ = True

    def init_algo_class(self) -> BaseOrientationMethod:
        return MahonyAHRS(use_magnetometer=True)

    def test_raises_if_no_mag(self):
        with pytest.raises(ValidationError, match=".*'mag_x', 'mag_y', 'mag_z'.*"):
            test_data = pd.DataFrame(columns=SF_COLS)
            test = self.init_algo_class()
            test.estimate(test_data, sampling_rate_hz=10)

    def test_mag_does_something(self):
        """Mahony should use the magnetometer to affect the heading estimate."""
        ori_with_y = np.array([0, 0, 0, 1.0])
        integral_error_y = np.zeros(3)
        for _i in range(50):
            ori_with_y, integral_error_y = _mahony_update_mag(
                np.array([1, 1, 0.0]),
                np.array([0, 0.0, 1.0]),
                np.array([0, 1.0, 0.0]),
                initial_orientation=ori_with_y,
                sampling_rate_hz=50,
                kp=1.0,
                ki=0.0,
                integral_error=integral_error_y,
            )

        ori_with_x = np.array([0, 0, 0, 1.0])
        integral_error_x = np.zeros(3)
        for _i in range(50):
            ori_with_x, integral_error_x = _mahony_update_mag(
                np.array([1, 1, 0.0]),
                np.array([0, 0.0, 1.0]),
                np.array([1.0, 0.0, 0.0]),
                initial_orientation=ori_with_x,
                sampling_rate_hz=50,
                kp=1.0,
                ki=0.0,
                integral_error=integral_error_x,
            )

        with pytest.raises(AssertionError):
            np.testing.assert_array_almost_equal(ori_with_x, ori_with_y, decimal=2)

    def test_single_stride_regression(self, snapshot):
        test = self.init_algo_class()
        data = get_magnetometer_l_path_data().iloc[:200]

        test.estimate(data, sampling_rate_hz=400)

        snapshot.assert_match(test.orientation_, test.__class__.__name__)
        rotated_data = test.rotated_data_
        rotated_data.columns.name = None
        snapshot.assert_match(rotated_data, f"{test.__class__.__name__}_rotated_data")
