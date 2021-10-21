import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA

from gaitmap.preprocessing.sensor_alignment import PcaAlignment
from gaitmap.utils.datatype_helper import MultiSensorData, get_multi_sensor_names


class TestPcaAlignment:
    """Test the function `align_to_gravity`."""

    sample_sensor_data: pd.DataFrame
    sample_sensor_dataset: MultiSensorData

    def test_single_sensor_input(self, healthy_example_imu_data):
        """Dummy test to see if the algorithm is generally working on the example data"""
        data = healthy_example_imu_data["left_sensor"]

        pca_align = PcaAlignment(pca_plane_axis=("gyr_x", "gyr_y"))
        pca_align = pca_align.align(data=data)

        assert len(pca_align.aligned_data_) == len(data)

        assert isinstance(pca_align.aligned_data_, pd.DataFrame)
        assert isinstance(pca_align.rotation_, Rotation)
        assert isinstance(pca_align.pca_, PCA)

    def test_multi_sensor_input(self, healthy_example_imu_data):
        """Dummy test to see if the algorithm is generally working on the example data"""
        data = healthy_example_imu_data

        pca_align = PcaAlignment(pca_plane_axis=("gyr_x", "gyr_y"))
        pca_align = pca_align.align(data=data)

        for sensor in get_multi_sensor_names(data):
            assert len(pca_align.aligned_data_[sensor]) == len(data[sensor])

            assert isinstance(pca_align.aligned_data_[sensor], pd.DataFrame)
            assert isinstance(pca_align.rotation_[sensor], Rotation)
            assert isinstance(pca_align.pca_[sensor], PCA)

    def test_multi_sensor_input(self, healthy_example_imu_data):
        """Dummy test to see if the algorithm is generally working on the example data"""
        data = healthy_example_imu_data

        pca_align = PcaAlignment(pca_plane_axis=("gyr_x", "gyr_y"))
        pca_align = pca_align.align(data=data)

        for sensor in get_multi_sensor_names(data):
            assert len(pca_align.aligned_data_[sensor]) == len(data[sensor])

            assert isinstance(pca_align.aligned_data_[sensor], pd.DataFrame)
            assert isinstance(pca_align.rotation_[sensor], Rotation)
            assert isinstance(pca_align.pca_[sensor], PCA)

    def test_invalid_pca_plane_axis(self, healthy_example_imu_data):
        """Test if value error is raised correctly if no static window can be found on dataset with given user
        settings."""
        data = healthy_example_imu_data

        with pytest.raises(ValueError, match=r".*Invalid axis for pca plane *"):
            PcaAlignment(pca_plane_axis=("abc", "gyr_y")).align(data)

        with pytest.raises(ValueError, match=r".*Invalid axis for pca plane *"):
            PcaAlignment(pca_plane_axis=("acc_x", "gyr_y")).align(data)

        with pytest.raises(ValueError, match=r".*Invalid axis for pca plane *"):
            PcaAlignment(pca_plane_axis=("acc_x")).align(data)

    def test_correct_rotation(self, healthy_example_imu_data):
        data = healthy_example_imu_data["left_sensor"]

        pca_align = PcaAlignment(pca_plane_axis=("gyr_x", "gyr_y"))
        pca_align = pca_align.align(data=data)

        expected_rot_matrix = np.array([[0.95948049, -0.28177506, 0.0], [0.28177506, 0.95948049, 0.0], [0.0, 0.0, 1.0]])

        assert_almost_equal(expected_rot_matrix, pca_align.rotation_.as_matrix())
