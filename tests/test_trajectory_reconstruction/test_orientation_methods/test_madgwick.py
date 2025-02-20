import numpy as np
import pandas as pd
import pytest

from gaitmap.base import BaseOrientationMethod, BaseType
from gaitmap.trajectory_reconstruction import MadgwickAHRS
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
