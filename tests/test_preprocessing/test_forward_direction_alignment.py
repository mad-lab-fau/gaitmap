import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from scipy.spatial.transform import Rotation

from gaitmap.base import BaseOrientationMethod, BasePositionMethod, BaseType
from gaitmap.preprocessing.sensor_alignment import ForwardDirectionSignAlignment
from gaitmap.utils.datatype_helper import get_multi_sensor_names
from gaitmap.utils.rotations import rotate_dataset
from tests.mixins.test_algorithm_mixin import TestAlgorithmMixin


class MetaTestConfig:
    algorithm_class = ForwardDirectionSignAlignment

    @pytest.fixture()
    def after_action_instance(self, healthy_example_imu_data) -> BaseType:
        fdsa = ForwardDirectionSignAlignment()
        fdsa.align(healthy_example_imu_data["left_sensor"].iloc[:1000], sampling_rate_hz=204.8)
        return fdsa


class TestMetaFunctionality(MetaTestConfig, TestAlgorithmMixin):
    __test__ = True


class TestForwardDirectionSignAlignment:
    """Test the forward direction sign alignment class `ForwardDirectionSignAlignment`."""

    def test_single_sensor_input(self, healthy_example_imu_data):
        """Dummy test to see if the algorithm is generally working on the example data"""
        data = healthy_example_imu_data["left_sensor"]

        fdsa = ForwardDirectionSignAlignment()
        fdsa = fdsa.align(data=data, sampling_rate_hz=204.8)

        assert len(fdsa.aligned_data_) == len(data)

        assert isinstance(fdsa.aligned_data_, pd.DataFrame)
        assert isinstance(fdsa.rotation_, Rotation)
        assert isinstance(fdsa.is_flipped_, bool)
        assert isinstance(fdsa.ori_method_, BaseOrientationMethod)
        assert isinstance(fdsa.pos_method_, BasePositionMethod)

    def test_multi_sensor_input(self, healthy_example_imu_data):
        """Dummy test to see if the algorithm is generally working on the example data"""
        data = healthy_example_imu_data

        fdsa = ForwardDirectionSignAlignment()
        fdsa = fdsa.align(data=data, sampling_rate_hz=204.8)

        for sensor in get_multi_sensor_names(data):
            assert len(fdsa.aligned_data_[sensor]) == len(data[sensor])

            assert isinstance(fdsa.aligned_data_[sensor], pd.DataFrame)
            assert isinstance(fdsa.rotation_[sensor], Rotation)
            assert isinstance(fdsa.is_flipped_[sensor], bool)
            assert isinstance(fdsa.ori_method_[sensor], BaseOrientationMethod)
            assert isinstance(fdsa.pos_method_[sensor], BasePositionMethod)

    def test_invalid_axis_combination(self):
        """Test if value error is raised correctly if invalid axis are defined."""

        with pytest.raises(ValueError, match=r".*Invalid rotation axis! *"):
            ForwardDirectionSignAlignment(forward_direction="x", rotation_axis="a").align(1, sampling_rate_hz=1)
        with pytest.raises(ValueError, match=r".*Invalid forward direction axis! *"):
            ForwardDirectionSignAlignment(forward_direction="a", rotation_axis="x").align(1, sampling_rate_hz=1)
        with pytest.raises(ValueError, match=r".*Invalid combination of rotation and forward direction axis! *"):
            ForwardDirectionSignAlignment(forward_direction="x", rotation_axis="x").align(1, sampling_rate_hz=1)

    def test_invalid_ori_method(self):
        """Test if value error is raised correctly if invalid ori_method class is passed."""

        with pytest.raises(ValueError, match=r".*The provided `ori_method` *"):
            ForwardDirectionSignAlignment(ori_method="abc").align(1, sampling_rate_hz=2)

    def test_invalid_pos_method(self):
        """Test if value error is raised correctly if invalid pos_method class is passed."""

        with pytest.raises(ValueError, match=r".*The provided `pos_method` *"):
            ForwardDirectionSignAlignment(pos_method="abc").align(1, sampling_rate_hz=2)

    def test_no_rotation(self, healthy_example_imu_data):
        """Test that no rotation is applied if the data is not rotated."""
        data = healthy_example_imu_data

        fwdsa = ForwardDirectionSignAlignment().align(data, sampling_rate_hz=204.8)

        for sensor in get_multi_sensor_names(data):
            assert_almost_equal(data[sensor].to_numpy(), fwdsa.aligned_data_[sensor].to_numpy())
            assert_almost_equal(np.rad2deg(fwdsa.rotation_[sensor].as_euler("zxy")), np.array([0.0, 0.0, 0.0]))

    def test_flip_rotation(self, healthy_example_imu_data):
        """Test that a correct 180flip is applied if the data is upside-down."""

        data = healthy_example_imu_data
        dataset_flipped = rotate_dataset(data, Rotation.from_euler("z", 180, degrees=True))

        fwdsa = ForwardDirectionSignAlignment().align(dataset_flipped, sampling_rate_hz=204.8)

        for sensor in get_multi_sensor_names(data):
            assert_almost_equal(data[sensor].to_numpy(), fwdsa.aligned_data_[sensor].to_numpy())
            assert_almost_equal(np.rad2deg(fwdsa.rotation_[sensor].as_euler("zxy")), np.array([180.0, 0.0, 0.0]))
